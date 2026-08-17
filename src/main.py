import anndata as ad


AIDA_FILE = "data/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad"

def main():
    adata_backed = ad.read_h5ad(AIDA_FILE, backed='r')

    print(adata_backed.n_obs)
    print(adata_backed.n_vars)

    adata_backed.file.close()

if __name__ == "__main__":
    main()