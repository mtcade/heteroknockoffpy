# stat.forest.hetero_gini.R
#
# Knockoff W statistics via ranger random forest with native categorical support.
#
# Mirrors the knockoff::stat.random_forest procedure but accepts mixed
# data.frames (numeric + factor columns) directly, using ranger's
# respect.unordered.factors = "partition" to handle factors without OHE.
#
# For factor columns, ranger's "impurity" importance is the total weighted
# decrease in node impurity (Gini for classification, variance for regression)
# across all splits on that variable — directly comparable to numeric importances.
#
# The column-wise random swap ensures the knockoff antisymmetry property:
# each column j is exchanged between X and X_k with probability 0.5.
#
# Usage via rpy2:
#   pkg <- rpy2.robjects.packages.STAP( r_code_string, "hetero_gini" )
#   W   <- pkg.stat_forest_hetero_gini( X_r, Xk_r, y_r, ... )
#   # W is a numeric vector of length p (one W statistic per original variable)


#' Knockoff W statistics for mixed data via ranger impurity importance.
#'
#' @param X     n x p data.frame with numeric and/or factor columns.
#' @param X_k   n x p data.frame of knockoffs, same schema as X.
#' @param y     Numeric vector or factor of length n (response variable).
#' @param ...   Additional arguments forwarded to ranger::ranger.
#' @return Numeric vector of length 2p: first p entries are original-X importances,
#'         last p are knockoff importances, both un-swapped to their true identity.
stat_forest_hetero_gini <- function( X, X_k, y, outcome.type = "continuous", ... ) {
    p    <- ncol( X )
    cols <- names( X )
    swap <- rbinom( p, 1, 0.5 )

    vargs <- list( ... )
    if ( !( "respect.unordered.factors" %in% names( vargs ) ) )
        vargs[[ "respect.unordered.factors" ]] <- "partition"
    if ( !( "min.node.size" %in% names( vargs ) ) && outcome.type %in% c( "categorical", "count" ) )
        vargs[[ "min.node.size" ]] <- 1L

    # -- Column-wise swap between X and X_k
    X.swap  <- X
    Xk.swap <- X_k
    for ( j in seq_len( p ) ) {
        if ( swap[ j ] ) {
            tmp              <- X.swap[[ j ]]
            X.swap[[ j ]]   <- Xk.swap[[ j ]]
            Xk.swap[[ j ]]  <- tmp
        }
    }

    # -- Build combined data.frame: original cols then knockoff cols (~)
    # Use the x/y ranger interface to avoid formula parsing issues with ~ names
    combined <- X.swap
    for ( col in cols )
        combined[[ paste0( col, "~" ) ]] <- Xk.swap[[ col ]]

    # -- Fit forest with impurity importance
    rf <- do.call(
        ranger::ranger,
        c(
            list(
                x             = combined,
                y             = y,
                importance    = "impurity",
                write.forest  = FALSE
            ),
            vargs
        )
    )

    Z <- as.vector( rf$variable.importance )   # length 2p

    # -- Un-swap so that positions 1..p are always original X importances
    # and p+1..2p are always knockoff importances, regardless of which columns
    # were exchanged before fitting.
    Z_orig  <- ifelse( swap == 1L, Z[ seq_len( p ) + p ], Z[ seq_len( p ) ] )
    Z_knock <- ifelse( swap == 1L, Z[ seq_len( p ) ],     Z[ seq_len( p ) + p ] )
    c( Z_orig, Z_knock )
}
