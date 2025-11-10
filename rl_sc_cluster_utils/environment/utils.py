"""Utility functions for the clustering environment."""

from anndata import AnnData


def validate_adata(adata: AnnData) -> None:
    """
    Validate AnnData object has required fields.

    This is a placeholder function that will be implemented in Stage 2
    when we have real data requirements.

    Required fields (to be implemented):
    - .obsm['X_scvi']: Embedding matrix
    - .uns['neighbors']: k-NN graph
    - .obs['clusters']: Initial cluster labels

    Parameters
    ----------
    adata : AnnData
        Annotated data object to validate

    Raises
    ------
    NotImplementedError
        This function is not yet implemented
    """
    raise NotImplementedError(
        "validate_adata() will be implemented in Stage 2 when data requirements are finalized"
    )
