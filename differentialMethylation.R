suppressMessages({
  library(limma)
})

# X: numeric matrix, rows = SAMPLES, cols = PROBES
# labels: character/factor vector (length = nrow(X))
# probe_names: character vector (length = ncol(X)) with probe IDs (e.g., cg0000...)
# top_k: number of top probes per class to keep
# robust: passed to eBayes()
runDM_multi <- function(X, labels, probe_names, top_k = 50, robust = TRUE) {
  labels <- as.factor(as.character(unlist(labels)))

  n_samples <- nrow(X); n_probes <- ncol(X)
  if (is.null(n_samples) || is.null(n_probes)) {
    stop("X must be a 2D numeric matrix with rows=samples, cols=probes.")
  }
  if (length(probe_names) != n_probes) {
    stop("probe_names length must equal ncol(X).")
  }

  # limma wants probes in rows, samples in cols
  X_t <- t(X)                                # rows = probes, cols = samples
  rownames(X_t) <- make.names(probe_names, unique = TRUE)  # set probe IDs (safe names)

  orig_levels <- levels(labels)
  safe_levels <- make.names(orig_levels)
  design <- model.matrix(~ 0 + labels)
  colnames(design) <- safe_levels

  keep_idx <- vapply(orig_levels, function(cl) sum(labels == cl) >= 2, logical(1))
  kept_orig <- orig_levels[keep_idx]
  kept_safe <- safe_levels[keep_idx]

  message(sprintf("X_t dims (probes x samples): %dx%d", nrow(X_t), ncol(X_t)))
  message(sprintf("Design dims: %dx%d", nrow(design), ncol(design)))
  message("Design colnames (safe): ", paste(colnames(design), collapse = ", "))

  if (length(kept_safe) == 0) {
    warning("No classes with >=2 samples; returning empty selection.")
    return(list(
      selected_union = character(0),
      per_class = setNames(vector("list", 0), character(0)),
      level_map = data.frame(orig = orig_levels, safe = safe_levels, stringsAsFactors = FALSE)
    ))
  }

  contrast_vec <- vapply(seq_along(kept_safe), function(i) {
    cl_safe <- kept_safe[i]
    others  <- kept_safe[-i]
    if (length(others) == 0) return(NA_character_)
    paste0(cl_safe, " - (", paste(others, collapse = " + "), ")/", length(others))
  }, character(1))
  contrast_vec <- contrast_vec[!is.na(contrast_vec)]

  fit  <- lmFit(X_t, design)
  C    <- makeContrasts(contrasts = contrast_vec, levels = design)
  fit2 <- contrasts.fit(fit, C)
  fit2 <- eBayes(fit2, robust = robust)

  per_class <- setNames(vector("list", length(kept_orig)), kept_orig)
  selected <- character(0)
  for (i in seq_along(contrast_vec)) {
    topi <- tryCatch({
      topTable(fit2, coef = i, number = Inf, sort.by = "P")
    }, error = function(e) {
      message("topTable error for coef ", i, ": ", conditionMessage(e))
      data.frame()
    })
    if (nrow(topi) > 0) {
      topi <- head(topi, top_k)
      genes <- rownames(topi)                # these are now your probe IDs
      per_class[[kept_orig[i]]] <- genes
      selected <- unique(c(selected, genes))
    } else {
      per_class[[kept_orig[i]]] <- character(0)
    }
  }

  message("Probes per contrast: ", paste(vapply(per_class, length, integer(1)), collapse = ", "))

  level_map <- data.frame(orig = orig_levels, safe = safe_levels, stringsAsFactors = FALSE)
  return(list(
    selected_union = selected,
    per_class = per_class,
    level_map = level_map
  ))
}