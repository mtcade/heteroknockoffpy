# scip.knockoffs.R
#
# Sequential Conditional Independence Procedure (SCIP) knockoff generation,
# entirely in R.
#
# Generates knockoffs sequentially: each column's knockoff is conditioned on
# all original variables plus all previously generated knockoffs.
#
# For categorical (factor) columns: fit a ranger probability forest, then
#   sample knockoff categories via row-wise softmax weighted draws.
# For numeric columns: fit a ranger regression forest, generate a knockoff
#   via conditional residual perturbation (normal draw or permutation).
#
# Exports two public functions (callable via rpy2 STAP):
#   scip.knockoffs( X, residuals.method, seed, ... )
#   scip.knockoffs_with_numeric( X, Xk.numeric, seed, ... )
#
# Note: rpy2 converts Python underscore kwargs to R dot params, so
#   residuals_method="normal"  ->  residuals.method = "normal"  in R.


# -- Private helpers

.scip.make_choices <- function( probs, categories ) {
    # Row-wise softmax sampling.
    # probs:      n x k non-negative matrix of raw probabilities.
    #             Rows need not sum to 1 — this function normalises them.
    # categories: character vector of length k, in the same column order as probs.
    # Returns:    factor of length n with levels == categories.
    row_sums <- rowSums( probs )
    row_sums[ row_sums == 0 ] <- 1.0   # guard against degenerate all-zero rows
    norm_probs <- probs / row_sums

    n       <- nrow( probs )
    k       <- length( categories )
    indices <- integer( n )
    for ( i in seq_len( n ) )
        indices[ i ] <- sample.int( k, size = 1L, prob = norm_probs[ i, ] )

    factor( categories[ indices ], levels = categories )
}


.scip.fit_probability_forest <- function( X, col, vargs ) {
    # Fit a ranger probability forest predicting factor column `col` from all
    # other columns in X (which may already contain knockoff columns).
    # Factor levels are sorted alphabetically before fitting so that the column
    # order of the returned probability matrix matches sort(levels(X[[col]])),
    # which corresponds to X[col].cat.get_categories().sort() in Python/polars.
    # Returns: n x k probability matrix.
    X_sorted        <- X
    X_sorted[[ col ]] <- factor( X[[ col ]], levels = sort( levels( X[[ col ]] ) ) )
    X_expl <- X[ , setdiff( names( X ), col ), drop = FALSE ]

    forest <- do.call(
        ranger::ranger,
        c(
            list(
                x           = X_expl,
                y           = X_sorted[[ col ]],
                probability = TRUE
            ),
            vargs
        )
    )
    predict( forest, data = X_expl )$predictions
}


.scip.fit_regression_forest <- function( X, col, vargs ) {
    # Fit a ranger regression forest predicting numeric column `col` from all
    # other columns in X (which may already contain knockoff columns).
    # Returns: numeric vector of length n (conditional expectations).
    X_expl <- X[ , setdiff( names( X ), col ), drop = FALSE ]

    forest <- do.call(
        ranger::ranger,
        c(
            list(
                x           = X_expl,
                y           = X[[ col ]],
                probability = FALSE
            ),
            vargs
        )
    )
    predict( forest, data = X_expl )$predictions
}


# -- Public functions

#' Full sequential SCIP knockoff generation for a mixed data.frame.
#'
#' Both categorical and numeric columns are processed in left-to-right order.
#' Each column's knockoff conditions on all original columns plus all
#' previously generated knockoffs (the scip_df grows column-by-column).
#'
#' @param X                n x p data.frame with numeric and/or factor columns.
#' @param residuals.method "normal" (default) or "permute" — how to resample
#'                         numeric conditional residuals.
#' @param seed             Integer RNG seed passed to set.seed() at entry.
#'                         Pass NULL (default) to use R's current RNG state.
#' @param ...              Additional arguments forwarded to ranger::ranger.
#' @return n x p data.frame of knockoffs with the same column names and types as X.
scip.knockoffs <- function(
    X,
    residuals.method = "normal",
    seed             = NULL,
    ...
){
    if ( is.character( X ) ) X <- as.data.frame( arrow::open_dataset( X ) )
    if ( !is.null( seed ) ) set.seed( as.integer( seed ) )

    vargs <- list( ... )
    if ( !( "respect.unordered.factors" %in% names( vargs ) ) )
        vargs[[ "respect.unordered.factors" ]] <- "partition"

    n       <- nrow( X )
    cols    <- names( X )
    scip_df <- X   # grows to include col~ knockoffs as we iterate

    for ( col in cols ) {
        ko <- paste0( col, "~" )

        if ( is.factor( X[[ col ]] ) ) {
            # -- Categorical knockoff: sample from probability forest
            probs      <- .scip.fit_probability_forest( scip_df, col, vargs )
            categories <- sort( levels( X[[ col ]] ) )
            scip_df[[ ko ]] <- .scip.make_choices( probs, categories )

        } else {
            # -- Numeric knockoff: conditional residual perturbation
            cond_exp  <- .scip.fit_regression_forest( scip_df, col, vargs )
            residuals <- X[[ col ]] - cond_exp   # use original X, not scip_df

            if ( residuals.method == "normal" ) {
                # sd() uses n-1 denominator, matching np.std(..., ddof=1)
                scip_df[[ ko ]] <- cond_exp + rnorm( n, mean = 0, sd = sd( residuals ) )

            } else if ( residuals.method == "permute" ) {
                scip_df[[ ko ]] <- cond_exp + sample( residuals )

            } else {
                stop( paste0( "Unrecognized residuals.method: ", residuals.method ) )
            }
        }
    }

    # Extract only the knockoff columns, renamed to original column names
    Xk          <- scip_df[ , paste0( cols, "~" ), drop = FALSE ]
    names( Xk ) <- cols
    Xk
}


#' Categorical-only SCIP knockoff generation (numeric knockoffs pre-provided).
#'
#' Numeric knockoffs are provided as a pre-built matrix \code{Xk.numeric}.
#' Only the factor columns are knocked out sequentially, conditioning on all
#' original columns + all numeric knockoffs + previously generated categorical
#' knockoffs.
#'
#' @param X           n x p data.frame with numeric and/or factor columns.
#' @param Xk.numeric  n x q numeric matrix of knockoffs for the non-factor
#'                    columns of X, in the same left-to-right order that
#'                    non-factor columns appear in X.
#' @param seed        Integer RNG seed passed to set.seed() at entry.
#' @param ...         Additional arguments forwarded to ranger::ranger.
#' @return n x p data.frame of knockoffs (all columns), same names/types as X.
scip.knockoffs_with_numeric <- function(
    X,
    Xk.numeric,
    seed = NULL,
    ...
){
    if ( is.character( X ) ) X <- as.data.frame( arrow::open_dataset( X ) )
    if ( !is.null( seed ) ) set.seed( as.integer( seed ) )

    vargs <- list( ... )
    if ( !( "respect.unordered.factors" %in% names( vargs ) ) )
        vargs[[ "respect.unordered.factors" ]] <- "partition"

    cols         <- names( X )
    numeric_cols <- cols[ !vapply( X, is.factor, logical( 1L ) ) ]
    n_numeric    <- length( numeric_cols )

    if ( ncol( Xk.numeric ) != n_numeric ) {
        stop( sprintf(
            "Xk.numeric has %d columns but X has %d non-factor columns",
            ncol( Xk.numeric ), n_numeric
        ) )
    }

    # Graft pre-built numeric knockoffs into scip_df with "~" suffix names
    scip_df <- X
    for ( j in seq_len( n_numeric ) )
        scip_df[[ paste0( numeric_cols[ j ], "~" ) ]] <- Xk.numeric[ , j ]

    # Iterate over original column order; skip non-factors
    for ( col in cols ) {
        if ( !is.factor( X[[ col ]] ) ) next

        ko         <- paste0( col, "~" )
        probs      <- .scip.fit_probability_forest( scip_df, col, vargs )
        categories <- sort( levels( X[[ col ]] ) )
        scip_df[[ ko ]] <- .scip.make_choices( probs, categories )
    }

    # Extract all knockoff columns in original column order, renamed
    Xk          <- scip_df[ , paste0( cols, "~" ), drop = FALSE ]
    names( Xk ) <- cols
    Xk
}
