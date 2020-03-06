import pandas as pd
import umap

def run_umap(encodings, ## assumed to be a dataframe or String
             metadata, ## assumed to be String
             save_path, ## assumed to be string
             ):  
    ### read in encodings if given as string path (otherwise: dataframe)
    if(type(encodings)==str or type(encodings)==file):
        encodings = pd.read_csv(encodings_path,header=None)
    
    ### read in all cell info
    if(type(metadata)==str or type(metadata)==file):
        mdata = pd.read_csv(metadata)
    
    ### fit UMAP
    reducer = umap.UMAP()
    umap_embedding_raw = reducer.fit_transform(encodings)
    umap_embedding_df = pd.DataFrame(umap_embedding_raw)
    umap_embedding_df.columns = ["d1","d2"]

    ### add UMAP coords to cell data
    umap_data = pd.concat([data,umap_embedding_df],axis=1)

    ### write csv OR return pd data frame
    if(save_path is not None):
        pd.DataFrame.to_csv(umap_data,
                            save_path)
    else:
        return(umap_data)

