# forest.ohe_probabilities.R
#
# Fit one ranger probability forest per factor column, predicting its category
# probabilities from all other columns, and return raw probability matrices as
# a named list.
#
# Usage via rpy2:
#   pkg <- rpy2.robjects.packages.STAP( r_code_string, "ohe_probs" )
#   result <- pkg.forest_ohe_probabilities( X, ... )
#   # result is a named list: factor column name -> n x k probability matrix
#   #   (column order matches sorted factor levels)
#
# Note: rpy2 converts Python underscore kwargs to R dot params.


#' Category probability matrices for all factor columns via ranger forests.
#'
#' @param X       n x p data.frame with a mix of numeric and factor columns.
#' @param ...     Additional arguments forwarded to ranger::ranger.
#' @return Named list mapping each factor column name to its n x k probability
#'         matrix, where k is the number of levels in that column and columns
#'         are in sorted level order.
forest.ohe_probabilities <- function( X, ... ){
    if ( is.character( X ) ) X <- as.data.frame( arrow::open_dataset( X ) )
    vargs <- list( ... )
    if ( !( "respect.unordered.factors" %in% names( vargs ) ) ){
        vargs[[ "respect.unordered.factors" ]] <- "partition"
    }

    factor_cols <- names( X )[ vapply( X, is.factor, logical( 1 ) ) ]

    result <- setNames( vector( "list", length( factor_cols ) ), factor_cols )

    for ( col in factor_cols ){
        X_explanatory <- X[ , setdiff( names( X ), col ), drop = FALSE ]

        # Reorder levels alphabetically so column order of returned matrix is
        # consistent with sorted levels (matching polars cat.get_categories().sort())
        X_sorted        <- X
        X_sorted[[ col ]] <- factor( X[[ col ]], levels = sort( levels( X[[ col ]] ) ) )

        forest <- do.call(
            ranger::ranger,
            c(
                list(
                    x           = X_explanatory,
                    y           = X_sorted[[ col ]],
                    probability = TRUE
                ),
                vargs
            )
        )
        result[[ col ]] <- predict( forest, data = X_explanatory )$predictions
    }

    result
}
