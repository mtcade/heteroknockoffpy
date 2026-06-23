# stat.forest.mald_count.R
#
# Count-outcome MALD importance using a ranger regression forest.
#
# For each predictor column the local gradient of the log expected value is
# computed (log of ranger regression predictions), which is the natural scale
# for multiplicative count models.
#
# Predictor handling:
#   Numeric  — bandwidth finite-difference of log-predicted-value
#   Factor   — max-minus-min sweep across predictor levels in log-predicted-value
#
# Usage via rpy2:
#   pkg <- rpy2.robjects.packages.STAP( r_code_string, "mald_count" )
#   importances <- pkg.stat_forest_mald_count( X_all, y, ... )
#
# Note: rpy2 converts Python underscore kwargs to R dot params, so
#   bandwidth_exponent=0.2  ->  bandwidth.exponent = 0.2  in R.

# -- Private helper

.mald_count.unzero <- function( vec ){
    # Replace zero (or negative) predicted values with the minimum positive value
    # before taking logs, to avoid -Inf.
    pos <- vec[ vec > 0 ]
    if ( length( pos ) == 0 ) return( rep( 1e-10, length( vec ) ) )
    min_pos <- min( pos )
    vec[ vec <= 0 ] <- min_pos
    vec
}


# -- Main function

#' Count-outcome MALD importance (regression forest on log expected value)
#'
#' @param X_all   n x 2p data.frame: original X and knockoff X_k concatenated column-wise.
#' @param y       Numeric (integer) vector of length n giving the count outcome.
#' @param bandwidth       Scale multiplier for numeric bandwidth (multiplied by sd(col)/n^exponent).
#' @param bandwidth.exponent  Exponent of sample size in bandwidth denominator.
#' @param exponent  Power applied to each pointwise absolute importance before averaging.
#' @param verbose   Print progress every `verbose` columns (0 = silent).
#' @param ...   Additional arguments forwarded to ranger::ranger.
#' @return Numeric vector of length 2p: mean absolute log-gradient importances.
stat.forest.mald_count <- function(
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
                x         = X_all,
                y         = y,
                probability = FALSE
            ),
            vargs
        )
    )

    # -- Base log-expected-value predictions  (n,)
    base.preds <- predict( forest, data = X_all )$predictions
    base.log   <- log( .mald_count.unzero( base.preds ) )

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
                "stat.forest.mald_count: column %d / %d", j, p_all
            ) )
        }

        if ( inherits( X_all[[ j ]], "factor" ) ){
            # -- Factor predictor: sweep over levels
            x.levels  <- levels( X_all[[ j ]] )
            k_x       <- length( x.levels )
            log_preds <- matrix( 0.0, nrow = n, ncol = k_x )

            for ( ki in seq_len( k_x ) ){
                X.test        <- X_all
                X.test[[ j ]] <- x.levels[[ ki ]]   # recycled to all n rows
                preds         <- predict( forest, data = X.test )$predictions
                log_preds[ , ki ] <- log( .mald_count.unzero( preds ) )
            }

            # Max-minus-min across predictor levels
            importances_pointwise[ , j ] <-
                apply( log_preds, 1, max ) - apply( log_preds, 1, min )

        } else {
            # -- Numeric predictor: bandwidth finite difference of log-predictions
            bw <- bandwidths[ j ]
            if ( bw == 0 ) next   # zero-variance column; importance stays 0

            X.test        <- X_all
            X.test[[ j ]] <- X_all[[ j ]] + bw

            mod.log <- log( .mald_count.unzero(
                predict( forest, data = X.test )$predictions
            ) )

            importances_pointwise[ , j ] <- ( mod.log - base.log ) / bw
        }
    }

    # Mean absolute importance with exponent
    colMeans( abs( importances_pointwise ) ^ exponent )
}
