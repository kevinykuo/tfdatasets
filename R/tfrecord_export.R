#' factory for creating functions that return Feature
create_make_feature <- function(type) {
  tf_type <- switch(
    type,
    integer = list("int64_list", tf$train$Int64List),
    double = list("float_list", tf$train$FloatList),
    list("bytes_list", tf$train$BytesList)
  ) %>%
    purrr::set_names(c("param", "constructor"))

  function(value) {
    force(value)
    value_list <- do.call(tf_type$constructor, list(value = as.list(value)))
    do.call(tf$train$Feature,
            purrr::set_names(list(value_list), tf_type$param))
  }
}

make_features <- function(l, feature_makers) {
  features_args <- purrr::map2(l, feature_makers, ~.y(.x)) %>%
    purrr::set_names(names(l))
  do.call(tf$train$Features, list(feature = features_args))
}

make_example <- function(l, feature_makers) {
  tf$train$Example(features = make_features(l, feature_makers))
}

make_sequence_example <- function(
  l, feature_makers, context_variables, sequence_variables) {
  context_features <- make_features(
    l[context_variables], feature_makers[context_variables])
  sequence_features <- make_feature_lists(
    l[sequence_variables], feature_makers[sequence_variables]
  )
  tf$train$SequenceExample(context = context_features,
                           feature_lists = sequence_features)
}

make_feature_lists <- function(l, feature_makers) {
  feature_lists <- purrr::map2(
    l, feature_makers,
    ~ tf$train$FeatureList(feature = lapply(.x, .y))
  )
  tf$train$FeatureLists(feature_list = feature_lists)
}

infer_feature_types <- function(l) {
  purrr::map_chr(l, purrr::compose(typeof, unlist))
}

#' @export
write_tfrecord <- function(
  data, path, record_type = c("Example", "SequenceExample")) {

  record_type <- rlang::arg_match(record_type)
  zipped <- purrr::transpose(data)
  variable_names <- names(data)

  # types of features, e.g. double, integer, character
  feature_types <- zipped[[1]] %>%
    infer_feature_types()
  feature_makers <- feature_types %>%
    map(create_make_feature)

  writer <- tf$python_io$TFRecordWriter(path)

  if (identical(record_type, "SequenceExample")) {
    sequence_variables <- zipped[[1]] %>%
      purrr::map_lgl(rlang::is_list) %>%
      `[`(variable_names, .)
    context_variables <- setdiff(variable_names, sequence_variables)
    purrr::walk(zipped, function(x) {
      example <- make_sequence_example(
        x, feature_makers, context_variables, sequence_variables)
      writer$write(example$SerializeToString())
    })
  } else {
    purrr::walk(zipped, function(x) {
      example <- make_example(x, feature_makers)
      writer$write(example$SerializeToString())
    })
  }

  writer$close()
  invisible(NULL)
}

#' @export
generate_parser <- function(data) {
  feature_names <- names(data)

  variable_lengths <- data %>%
    purrr::keep(is_bare_list) %>%
    purrr::map(~ purrr::map_int(.x, length) %>%
                 unique()
    ) %>%
    purrr::map_int(~ (
      if (rlang::is_scalar_integer(.x)) .x else NA_integer_
    ))

  # whether each feature has fixed or variable length
  feature_shapes <- ifelse(feature_names %in% names(variable_lengths),
                           variable_lengths, 1L)

  # types of features, e.g. double, intger, character
  feature_types <- purrr::transpose(data)[[1]] %>%
    infer_feature_types()

  create_parse_function(feature_names, feature_shapes, feature_types)
}

create_parse_function <- function(feature_names, feature_shapes, feature_types) {
  make_feature_config <- function(feature_shape, feature_type) {
    feature_type <- switch(
      feature_type,
      integer = tf$int64,
      double = tf$float32,
      tf$string
    )

    if (is.na(feature_shape)) {
      tf$VarLenFeature(feature_type)
    } else {
      tf$FixedLenFeature(shape(feature_shape), feature_type)
    }
  }

  features <- purrr::map2(feature_shapes, feature_types, make_feature_config) %>%
    purrr::set_names(feature_names)

  function(example_proto) {
    force(example_proto)
    tf$parse_single_example(
      serialized = example_proto,
      features = features)
  }
}
