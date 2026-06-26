# stat.forest.mald_categorical.R
#
# Categorical-outcome MALD importance using a ranger probability forest.
#
# For each predictor column the local gradient of the log-probability vector
# is computed, then contrasted against the first outcome category, and the
# Mahalanobis norm is taken using the Fisher information of the base log-odds.
# This mirrors the approach in stat.forest.local_grad.R but for a categorical
# outcome (probability = TRUE forest) rather than a continuous/count outcome.
#
# Predictor handling:
#   Numeric  — bandwidth finite-difference of log-prob vector
#   Factor   — max-minus-min sweep across predictor levels
#
# Usage via rpy2:
#   pkg <- rpy2.robjects.packages.STAP( r_code_string, "mald_cat" )
#   importances <- pkg.stat_forest_mald_categorical( X_all, y, ... )
#
# Note: rpy2 converts Python underscore kwargs to R dot params, so
#   bandwidth_exponent=0.2  ->  bandwidth.exponent = 0.2  in R.

# -- Private helpers (prefixed to avoid global namespace collisions)

.mald_cat.unzero_normalize <- function( mat ){
    # Replace zero cells with the minimum nonzero value, then row-normalize.
    min_nonzero <- min( mat[ mat > 0 ] )
    mat[ mat == 0 ] <- min_nonzero
    mat / rowSums( mat )
}

.mald_cat.mahal_norms <- function( contrasts, VI ){
    # Mahalanobis norm from the origin for each row of `contrasts` (n x d).
    # Computed as sqrt( rowSums( (contrasts %*% VI) * contrasts ) ).
    # More efficient than apply + t(v) %*% VI %*% v for large n.
    sqrt( rowSums( ( contrasts %*% VI ) * contrasts ) )
}


# -- Main function

#' Categorical-outcome MALD importance (probability forest)
#'
#' @param X_all   n x 2p data.frame: original X and knockoff X_k concatenated column-wise.
#' @param y       Factor vector of length n giving the outcome category for each observation.
#' @param bandwidth       Scale multiplier for numeric bandwidth (multiplied by sd(col)/n^exponent).
#' @param bandwidth.exponent  Exponent of sample size in bandwidth denominator.
#' @param exponent  Power applied to each pointwise importance before averaging.
#' @param verbose   Print progress every `verbose` columns (0 = silent).
#' @param ...   Additional arguments forwarded to ranger::ranger.
#' @return Numeric vector of length 2p: mean absolute local-gradient importances.
stat.forest.mald_categorical <- function(
    X_all,
    y,
    bandwidth           = 1,
    bandwidth.exponent  = 0.2,
    exponent            = 2,
    verbose             = 0,
    ...
){
    if ( inherits( y, "data.frame" ) ){
        y <- y[[ 1 ]]
    }
    if ( !is.factor( y ) ){
        y <- as.factor( y )
    }

    vargs <- list( ... )
    if ( !( "respect.unordered.factors" %in% names( vargs ) ) ){
        vargs[[ "respect.unordered.factors" ]] <- "partition"
    }

    n     <- nrow( X_all )
    p_all <- ncol( X_all )
    n.factor <- n ^ bandwidth.exponent

    # -- Fit probability forest
    forest <- do.call(
        ranger::ranger,
        c(
            list(
                x           = X_all,
                y           = y,
                probability = TRUE
            ),
            vargs
        )
    )

    # -- Base log-probability predictions  (n x k_y)
    base.probs <- predict( forest, data = X_all )$predictions
    base.log   <- log( .mald_cat.unzero_normalize( base.probs ) )
    k_y        <- ncol( base.log )

    # Only one class predicted — log-odds contrasts are undefined; return zeros.
    if ( k_y < 2 ){
        warning( "stat.forest.mald_categorical: only one class predicted; returning zero importances" )
        return( rep( 0.0, p_all ) )
    }

    # -- Fisher information: inv-cov of log-odds contrasts  (k_y-1 x k_y-1)
    # Contrast each log-prob column against the first category
    base.contrasts <- as.matrix(
        base.log[ , 2:k_y, drop = FALSE ] - base.log[ , 1 ]
    )   # n x (k_y-1)
    VI <- solve( cov( base.contrasts ) )

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
                "stat.forest.mald_categorical: column %d / %d", j, p_all
            ) )
        }

        if ( inherits( X_all[[ j ]], "factor" ) ){
            # -- Factor predictor: sweep over levels
            x.levels <- levels( X_all[[ j ]] )
            k_x      <- length( x.levels )

            # Collect per-level log-odds contrasts (vs first outcome class): n x (k_y-1) x k_x
            logodds_per_level <- array( 0.0, dim = c( n, k_y - 1, k_x ) )
            for ( ki in seq_len( k_x ) ){
                X.test        <- X_all
                X.test[[ j ]] <- x.levels[[ ki ]]   # recycled to all n rows
                preds         <- predict( forest, data = X.test )$predictions
                log_p         <- log( .mald_cat.unzero_normalize( preds ) )  # n x k_y
                logodds_per_level[ , , ki ] <- log_p[ , 2:k_y, drop = FALSE ] - log_p[ , 1 ]
            }

            # Range of log-odds contrasts across predictor levels  ->  n x (k_y-1)
            contrasts <- as.matrix(
                apply( logodds_per_level, c( 1, 2 ), max ) -
                apply( logodds_per_level, c( 1, 2 ), min )
            )

        } else {
            # -- Numeric predictor: bandwidth finite difference
            bw <- bandwidths[ j ]
            if ( bw == 0 ) next   # zero-variance column; importance stays 0

            X.test        <- X_all
            X.test[[ j ]] <- X_all[[ j ]] + bw

            mod.log <- log( .mald_cat.unzero_normalize(
                predict( forest, data = X.test )$predictions
            ) )

            # Local gradient of log-probs  ->  n x k_y
            grad_mat  <- ( mod.log - base.log ) / bw

            # Log-odds contrast  ->  n x (k_y-1)
            contrasts <- as.matrix(
                grad_mat[ , 2:k_y, drop = FALSE ] - grad_mat[ , 1 ]
            )
        }

        importances_pointwise[ , j ] <- .mald_cat.mahal_norms( contrasts, VI )
    }

    # Mean importance with exponent (Mahalanobis norms are already non-negative)
    colMeans( importances_pointwise ^ exponent )
}
