# forest.conditional_expectations.R
#
# Fit one ranger regression forest per non-factor column, predicting it from
# all other columns, and return in-sample conditional expectations.
#
# Usage via rpy2:
#   pkg <- rpy2.robjects.packages.STAP( r_code_string, "cond_exp" )
#   result <- pkg.forest_conditional_expectations( X, ... )
#   # result is a named list: column name -> numeric vector of predictions (length n)
#
# Note: rpy2 converts Python underscore kwargs to R dot params.


#' Conditional expectations for all numeric columns via ranger forests.
#'
#' @param X       n x p data.frame with a mix of numeric and factor columns.
#' @param ...     Additional arguments forwarded to ranger::ranger.
#' @return Named list mapping each non-factor column name to its n-vector of
#'         in-sample conditional expectation predictions.
forest.conditional_expectations <- function( X, ... ){
    if ( is.character( X ) ) X <- as.data.frame( arrow::open_dataset( X ) )
    vargs <- list( ... )
    if ( !( "respect.unordered.factors" %in% names( vargs ) ) ){
        vargs[[ "respect.unordered.factors" ]] <- "partition"
    }

    numeric_cols <- names( X )[ !vapply( X, is.factor, logical( 1 ) ) ]

    result <- setNames( vector( "list", length( numeric_cols ) ), numeric_cols )

    for ( col in numeric_cols ){
        X_explanatory <- X[ , setdiff( names( X ), col ), drop = FALSE ]
        forest <- do.call(
            ranger::ranger,
            c(
                list(
                    x           = X_explanatory,
                    y           = X[[ col ]],
                    probability = FALSE
                ),
                vargs
            )
        )
        result[[ col ]] <- predict( forest, data = X_explanatory )$predictions
    }

    result
}
