make_feature <- function(value) {
  feature_args <- switch(
    typeof(value),
    integer = list(int64_list = tf$train$Int64List(value = as.list(value))),
    double = list(float_list = tf$train$FloatList(value = as.list(value))),
    list(bytes_list = tf$train$BytesList(value = as.list(value)))
  )
  do.call(tf$train$Feature, feature_args)
}

make_features <- function(l) {
  features_args <- map(l, make_feature) %>%
    purrr::set_names(names(l))
  do.call(tf$train$Features, list(feature = features_args))
}

make_example <- function(l) {
  tf$train$Example(features = make_features(l))
}
