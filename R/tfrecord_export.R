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

write_tfrecord <- function(data, path) {
  l <- purrr::transpose(data)

  # types of features, e.g. double, intger, character
  feature_types <- l[[1]] %>%
    purrr::map_chr(purrr::compose(typeof, unlist))

  feature_makers <- feature_types %>%
    map(create_make_feature)

  writer <- tf$python_io$TFRecordWriter(path)
  purrr::walk(l, function(x) {
    example <- make_example(x, feature_makers)
    writer$write(example$SerializeToString())
  })
  writer$close()
  invisible(NULL)
}

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

  l <- purrr::transpose(data)

  # types of features, e.g. double, intger, character
  feature_types <- l[[1]] %>%
    purrr::map_chr(purrr::compose(typeof, unlist))

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
