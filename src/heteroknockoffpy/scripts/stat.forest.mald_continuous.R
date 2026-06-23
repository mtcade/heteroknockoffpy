# stat.forest.mald_continuous.R
#
# Continuous-outcome MALD importance using a ranger regression forest.
#
# For each predictor column the local gradient of the predicted value is
# computed, which is the natural scale for continuous outcomes.
#
# Predictor handling:
#   Numeric  — bandwidth finite-difference of predicted value
#   Factor   — max-minus-min sweep across predictor levels of predicted value
#
# Usage via rpy2:
#   pkg <- rpy2.robjects.packages.STAP( r_code_string, "mald_continuous" )
#   importances <- pkg.stat_forest_mald_continuous( X_all, y, ... )
#
# Note: rpy2 converts Python underscore kwargs to R dot params, so
#   bandwidth_exponent=0.2  ->  bandwidth.exponent = 0.2  in R.


#' Continuous-outcome MALD importance (regression forest)
#'
#' @param X_all   n x 2p data.frame: original X and knockoff X_k concatenated column-wise.
#' @param y       Numeric vector of length n giving the continuous outcome.
#' @param bandwidth       Scale multiplier for numeric bandwidth (multiplied by sd(col)/n^exponent).
#' @param bandwidth.exponent  Exponent of sample size in bandwidth denominator.
#' @param exponent  Power applied to each pointwise absolute importance before averaging.
#' @param verbose   Print progress every `verbose` columns (0 = silent).
#' @param ...   Additional arguments forwarded to ranger::ranger.
#' @return Numeric vector of length 2p: mean absolute importances.
stat.forest.mald_continuous <- function(
    X_all,
    y,
    bandwidth           = 1,
    bandwidth.exponent  = 0.2,
    exponent            = 1,
    verbose             = 0,
    ...
){
    if ( inherits( y, "data.frame" ) ){
        y <- y[[ 1 ]]
    }
    y <- as.numeric( y )

    vargs <- list( ... )
    if ( !( "respect.unordered.factors" %in% names( vargs ) ) ){
        vargs[[ "respect.unordered.factors" ]] <- "partition"
    }

    n        <- nrow( X_all )
    p_all    <- ncol( X_all )
    n.factor <- n ^ bandwidth.exponent

    # -- Fit regression forest
    forest <- do.call(
        ranger::ranger,
        c(
            list(
                x           = X_all,
                y           = y,
                probability = FALSE
            ),
            vargs
        )
    )

    # -- Base predictions  (n,)
    base.preds <- predict( forest, data = X_all )$predictions

    # -- Per-column bandwidths for numeric predictors
    bandwidths <- vapply(
        X_all,
        function( col ) if ( is.numeric( col ) ) sd( col ) * bandwidth / n.factor else 0,
        numeric( 1 )
    )

    # -- Pointwise importance matrix  (n x p_all)
    importances_pointwise <- matrix( 0.0, nrow = n, ncol = p_all )

    for ( j in seq_len( p_all ) ){
        if ( verbose > 0 && j %% verbose == 0 ){
            message( sprintf(
                "stat.forest.mald_continuous: column %d / %d", j, p_all
            ) )
        }

        if ( inherits( X_all[[ j ]], "factor" ) ){
            # -- Factor predictor: sweep over levels
            x.levels  <- levels( X_all[[ j ]] )
            k_x       <- length( x.levels )
            preds_mat <- matrix( 0.0, nrow = n, ncol = k_x )

            for ( ki in seq_len( k_x ) ){
                X.test        <- X_all
                X.test[[ j ]] <- x.levels[[ ki ]]   # recycled to all n rows
                preds_mat[ , ki ] <- predict( forest, data = X.test )$predictions
            }

            # Max-minus-min across predictor levels
            importances_pointwise[ , j ] <-
                apply( preds_mat, 1, max ) - apply( preds_mat, 1, min )

        } else {
            # -- Numeric predictor: bandwidth finite difference
            bw <- bandwidths[ j ]
            if ( bw == 0 ) next   # zero-variance column; importance stays 0

            X.test        <- X_all
            X.test[[ j ]] <- X_all[[ j ]] + bw

            importances_pointwise[ , j ] <-
                ( predict( forest, data = X.test )$predictions - base.preds ) / bw
        }
    }

    # Mean absolute importance with exponent
    colMeans( abs( importances_pointwise ) ^ exponent )
}
