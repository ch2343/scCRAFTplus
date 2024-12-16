import numpy as np
import pandas as pd
from scipy.sparse import issparse


import torch 
def rank_genes(adata, cell_types_markers):
    """
    Rank all genes that appear in the dictionary for all cell types.

    Parameters:
    adata (anndata.AnnData): The annotated data matrix.
    cell_types_markers (dict): A dictionary mapping cell types to lists of markers.

    Returns:
    pd.DataFrame: DataFrame of gene ranks.
    """
    # Flatten the marker lists and get unique markers
    all_markers = set([marker for markers in cell_types_markers.values() for marker in markers])
    available_markers = [marker for marker in all_markers if marker in adata.var_names]

    if not available_markers:
        raise ValueError("No available markers found in the dataset.")

    # Extract the expression data for available markers
    marker_data = adata[:, available_markers].X

    # Convert to DataFrame and rank the genes for each cell
    if isinstance(marker_data, np.ndarray):
        ranks = pd.DataFrame(marker_data, columns=available_markers).rank(method='average')
    else:  # Handle sparse matrix
        ranks = pd.DataFrame(marker_data.toarray(), columns=available_markers).rank(method='average')

    return ranks

def calculate_u_scores_from_ranks(ranks, cell_type, markers):
    """
    Calculate U scores for the specified cell type based on precomputed ranks.

    :param ranks: DataFrame of precomputed ranks.
    :param cell_type: Name of the cell type for which to calculate U scores.
    :param markers: List of signature genes for the cell type.
    :return: Series of U scores for each cell in the dataset.
    """
    # Filter markers that exist in the ranks DataFrame
    available_markers = [marker for marker in markers if marker in ranks.columns]
    
    if not available_markers:
        print(f"No available markers found for cell type {cell_type} in the dataset.")
        return None

    # Extract the rank data for available markers
    signature_ranks = ranks[available_markers]

    # Calculate the U score for each cell
    n = len(available_markers)
    u_values = signature_ranks.sum(axis=1) - n * (n + 1) / 2
    u_values = u_values / n
    u_scores = u_values / u_values.max() - u_values.min() / u_values.max()

    return u_scores

def process_cell_types_with_ranks(adata, cell_types_markers, ranks, marker_label = 'marker_label'):
    """
    Processes each cell type in an AnnData object, computing U scores using precomputed ranks and updating marker labels.

    Parameters:
    adata (anndata.AnnData): The annotated data matrix.
    cell_types_markers (dict): A dictionary mapping cell types to lists of markers.
    ranks (pd.DataFrame): DataFrame of precomputed ranks.

    Returns:
    None, modifies `adata` in place.
    """
    # Initialize marker labels with NaN
    adata.obs[marker_label] = np.nan

    # Calculate U scores for each cell type and update adata.obs
    for cell_type, markers in cell_types_markers.items():
        u_scores = calculate_u_scores_from_ranks(ranks, cell_type, markers)
        if u_scores is not None and not u_scores.isnull().all():
            adata.obs[f'{cell_type}_U_score'] = u_scores.values
            print(f"Processed {cell_type}")
        else:
            print(f"Skipped {cell_type} due to no available markers or all NaN scores")

    # Identify the column of the highest U score for each cell and assign as marker_label
    u_score_columns = [col for col in adata.obs.columns if col.endswith('_U_score')]
    adata.obs[marker_label] = adata.obs[u_score_columns].idxmax(axis=1).str.replace('_U_score', '')
    cell_types = sorted(cell_types_markers.keys())  # Assuming `cell_types_markers` is a dictionary mapping cell types to their markers
    adata.obs[marker_label] = pd.Categorical(adata.obs[marker_label], categories=cell_types)

    # Convert the DataFrame of U scores to a numpy array and store in adata.obsm
    u_scores_matrix = adata.obs[u_score_columns]
    adata.obsm['U_scores'] = u_scores_matrix.values

def softmax(x, axis=1):
    """
    Compute softmax values for each set of scores in x along the specified axis.
    
    Parameters:
    x (np.ndarray): Array of scores.
    axis (int): Axis along which the softmax is computed.
    
    Returns:
    np.ndarray: Softmax probabilities.
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # Stability improvement
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
def perform_permutation_tests(adata, cell_types_markers, ranks, n_permutations=100, temperature=1):
    """
    Perform permutation tests to identify unknown cell types.

    Parameters:
    adata (anndata.AnnData): The annotated data matrix.
    cell_types_markers (dict): A dictionary mapping cell types to lists of markers.
    ranks (pd.DataFrame): DataFrame of precomputed ranks.
    n_permutations (int): Number of permutations to perform.
    temperature (float): Temperature parameter for energy calculation.

    Returns:
    None, modifies `adata` in place.
    """
    # Precompute original U scores and energies
    original_u_scores_matrix = []
    for cell_type, markers in cell_types_markers.items():
        u_scores = calculate_u_scores_from_ranks(ranks, cell_type, markers)
        if u_scores is not None:
            original_u_scores_matrix.append(u_scores.values)
    
    original_u_scores_matrix = np.array(original_u_scores_matrix).T
    original_softmax_probs = softmax(original_u_scores_matrix, axis=1)
    original_energies = -temperature * torch.logsumexp(torch.tensor(original_softmax_probs) / temperature, dim=1).numpy()

    # Placeholder for permutation energies
    permuted_energies = np.zeros((n_permutations, adata.n_obs))

    # Store the original column names
    original_columns = ranks.columns.tolist()

    for i in range(n_permutations):
        # Permute the column names
        permuted_columns = np.random.permutation(original_columns)
        
        # Apply the permuted column names
        permuted_ranks = ranks.copy()
        permuted_ranks.columns = permuted_columns

        # Calculate U scores for the permuted ranks
        permuted_u_scores_matrix = []
        for cell_type, markers in cell_types_markers.items():
            permuted_u_scores = calculate_u_scores_from_ranks(permuted_ranks, cell_type, markers)
            if permuted_u_scores is not None:
                permuted_u_scores_matrix.append(permuted_u_scores.values)

        if not permuted_u_scores_matrix:
            continue  # Skip if no U scores were calculated for this permutation

        permuted_u_scores_matrix = np.array(permuted_u_scores_matrix).T

        # Apply softmax to permuted U scores
        permuted_softmax_probs = softmax(permuted_u_scores_matrix, axis=1)

        # Calculate permuted energies
        permuted_energies[i, :] = -temperature * torch.logsumexp(torch.tensor(permuted_softmax_probs) / temperature, dim=1).numpy()

    # Compare permuted energies with original energies
    exceed_count = (permuted_energies < original_energies).sum(axis=0)
    adata.obs['unknown_label'] = exceed_count > (n_permutations * 0.05)  # Adjust the threshold accordingly
    adata.obs['hard_label'] = adata.obs['marker_label']
    # Add "Unknown" as a category in 'marker_label'
    adata.obs['marker_label'] = adata.obs['marker_label'].cat.add_categories(['Unknown'])

    # Update marker_label to "Unknown" for cells identified as unknown
    adata.obs.loc[adata.obs['unknown_label'], 'marker_label'] = 'Unknown'

    # Convert the 'marker_label' to categorical type with "Unknown" as one of the categories
    cell_types = sorted(cell_types_markers.keys())
    categories = cell_types + ['Unknown']
    adata.obs['marker_label'] = pd.Categorical(adata.obs['marker_label'], categories=categories)
    adata.obs['unknown_label'] = adata.obs['unknown_label'].astype('category')


markers_pancreas = {
    'cycling': ['UBE2C', 'TOP2A', 'CDK1', 'BIRC5', 'PBK', 'CDKN3', 'MKI67', 'CDC20', 'CCNB2', 'CDCA3'],
    'immune': ['ACP5', 'APOE', 'HLA-DRA', 'TYROBP', 'LAPTM5', 'SDS', 'FCER1G', 'C1QC', 'C1QB', 'SRGN'],
    'quiescent_stellate': ['RGS5', 'C11orf96', 'FABP4', 'CSRP2', 'IL24', 'ADIRF', 'NDUFA4L2', 'GPX3', 'IGFBP4', 'ESAM'],
    'endothelial': ['PLVAP', 'RGCC', 'ENG', 'PECAM1', 'ESM1', 'SERPINE1', 'CLDN5', 'STC1', 'MMP1', 'GNG11'],
    'schwann': ['NGFR', 'CDH19', 'UCN2', 'SOX10', 'S100A1', 'PLP1', 'TSPAN11', 'WNT16', 'SOX2', 'TFAP2A'],
    'activated_stellate': ['COL1A1', 'COL1A2', 'COL6A3', 'COL3A1', 'TIMP3', 'TIMP1', 'CTHRC1', 'SFRP2', 'BGN', 'LUM'],
    'epsilon': ['BHMT', 'VSTM2L', 'PHGR1', 'TM4SF5', 'ANXA13', 'ASGR1', 'DEFB1', 'GHRL', 'COL22A1', 'OLFML3'],
    'gamma': ['PPY', 'AQP3', 'MEIS2', 'ID2', 'GPC5-AS1', 'CARTPT', 'PRSS23', 'ETV1', 'PPY2', 'TUBB2A'],
    'delta': ['SST', 'RBP4', 'SERPINA1', 'RGS2', 'PCSK1', 'SEC11C', 'HHEX', 'LEPR', 'MDK', 'LY6H'],
    'ductal': ['SPP1', 'MMP7', 'IGFBP7', 'KRT7', 'ANXA4', 'SERPINA1', 'LCN2', 'CFTR', 'KRT19', 'SERPING1'],
    'acinar': ['REG1A', 'PRSS1', 'CTRB2', 'CTRB1', 'REG1B', 'CELA3A', 'PRSS2', 'REG3A', 'CPA1', 'CLPS'],
    'beta': ['IAPP', 'INS', 'DLK1', 'INS-IGF2', 'G6PC2', 'HADH', 'ADCYAP1', 'GSN', 'NPTX2', 'C12orf75'],
    'alpha': ['GCG', 'TTR', 'PPP1R1A', 'CRYBA2', 'TM4SF4', 'MAFB', 'GC', 'GPX3', 'PCSK2', 'PEMT']
}


markers_lung = {
    'Alveolar_Epithelial_Type_1': ['EMP2', 'AGER', 'GPRC5A', 'LMO7', 'RTKN2', 'ARHGEF26', 'LAMA3', 'DST', 'SPOCK2', 'SCEL'],
    'B': ['MS4A1', 'BANK1', 'LINC00926', 'CD79A', 'BLK', 'CD79B', 'VPREB3', 'SP140', 'CD22', 'ADAM28'],
    'Basal': ['KRT15', 'S100A2', 'KRT17', 'KRT5', 'KRT19', 'PERP', 'CLDN1', 'FHL2', 'DST', 'ITGA2'],
    'CD14_Monocyte': ['S100A9', 'S100A12', 'S100A8', 'VCAN', 'FCN1', 'CD300E', 'CTSS', 'CD99', 'CSF3R', 'METTL9'],
    'CD4_T': ['CD2', 'IL7R', 'TRAT1', 'BCL11B', 'ITK', 'CAMK4', 'CD6', 'LEF1', 'MALAT1', 'MAL'],
    'Ciliated': ['RSPH1', 'CAPS', 'FAM92B', 'CDHR3', 'CCDC170', 'SNTN', 'C20orf85', 'ZMYND10', 'DNAH6', 'FANK1'],
    'Dendritic': ['HLA-DPB1', 'FCGR2B', 'HLA-DMB', 'CLEC10A', 'CSF2RA', 'MS4A6A', 'HLA-DQA1', 'RGS1', 'GPR183', 'FPR3'],
    'Fibroblast': ['FBLN1', 'ADH1B', 'DCN', 'LUM', 'C1S', 'C1R', 'SLIT2', 'C3', 'CFD', 'ABLIM1'],
    'Lymphatic': ['CCL21', 'MMRN1', 'PROX1', 'PPFIBP1', 'TFF3', 'TBX1', 'PKHD1L1', 'IGFBP7', 'TFPI', 'TIMP3'],
    'Macrophage': ['ACP5', 'C1QA', 'C1QB', 'VSIG4', 'MARCO', 'OLR1', 'APOC1', 'SLCO2B1', 'GPD1', 'ITGAX'],
    'Natural_Killer': ['PRF1', 'NKG7', 'KLRF1', 'CD247', 'GNLY', 'PYHIN1', 'KLRD1', 'SPON2', 'FCGR3A', 'GZMB'],
    'Plasma': ['MZB1', 'DERL3', 'SEC11C', 'XBP1', 'TNFRSF17', 'SSR4', 'POU2AF1', 'HSP90B1', 'PRDX4', 'MANF'],
    'Alveolar_Epithelial_Type_2': ['SFTPA1', 'SFTPA2', 'NAPSA', 'SFTPB', 'SFTPD', 'SFTPC', 'SLC34A2', 'ABCA3', 'LAMP3', 'LPCAT1'],
    'Basophil_Mast': ['KIT', 'CPA3', 'TPSAB1', 'HPGDS', 'VWA5A', 'MS4A2', 'HDC', 'CD44', 'GATA2', 'GRAP2'],
    'CD16_Monocyte': ['LST1', 'LILRB2', 'AIF1', 'COTL1', 'SAT1', 'FCN1', 'FCGR3A', 'RPS19', 'RHOC', 'IFITM2'],
    'Capillary': ['FCN3', 'EPAS1', 'RAMP3', 'CDH5', 'PODXL', 'BTNL9', 'NOSTRIN', 'FLNB', 'CDC42EP1', 'CD81'],
    'CD8_T': ['CCL5', 'CD3D', 'NKG7', 'GZMH', 'CD3G', 'CD8A', 'GZMA', 'CD8B', 'RPS27', 'RPL34'],
    'Club': ['SCGB3A2', 'CYB5A', 'SFTPB', 'PIGR', 'C16orf89', 'CTSE', 'KIAA1324', 'CEACAM6', 'WFDC2', 'DSTN'],
    'Myofibroblast': ['COL1A1', 'COL1A2', 'COL3A1', 'COL6A3', 'ASPN', 'LTBP1', 'ITGBL1', 'CALD1', 'POSTN', 'BGN'],
    'Plasmacytoid_Dendritic': ['PPP1R14B', 'IRF4', 'GZMB', 'IL3RA', 'IRF7', 'PLD4', 'SEC61B', 'PTPRE', 'NPC2', 'TYROBP'],
    'Proliferating_Macrophage': ['H2AFZ', 'CDK1', 'UBE2C', 'ANLN', 'BIRC5', 'NUSAP1', 'STMN1', 'TPX2', 'TYMS', 'KPNA2'],
    'Proliferating_NK_T': ['NUSAP1', 'TOP2A', 'HMGB2', 'BIRC5', 'H2AFV', 'SMC4', 'CENPF', 'AURKB', 'UBE2C', 'KIF2C'],
    'Bronchial_Vessel': ['PLVAP', 'ZNF385D', 'SPRY1', 'POSTN', 'SPARCL1', 'VWF', 'EMCN', 'COL15A1', 'INHBB', 'ENPP2'],
    'Mucous': ['SCGB3A1', 'BPIFB1', 'SLPI', 'LCN2', 'WFDC2', 'SAA1', 'SCGB1A1', 'RARRES1', 'MUC5B', 'MSMB'],
    'Vein': ['VWF', 'SELP', 'SLCO2A1', 'LIFR', 'PLA1A', 'PTPRB', 'PRSS23', 'CPE', 'TGM2', 'SRPX'],
    'Pericyte': ['COX4I2', 'PDGFRB', 'TACC1', 'CALD1', 'NID1', 'COL4A1', 'COL4A2', 'LAMB1', 'LAMC3', 'CHN1'],
    'Smooth_Muscle': ['TPM2', 'ACTA2', 'MYL9', 'MYH11', 'CALD1', 'DSTN', 'CNN1', 'DES', 'PLN', 'CSRP1'],
    'Artery': ['DKK2', 'PTPRB', 'EFNB2', 'EPAS1', 'GJA5', 'HEY1', 'ARGLU1', 'FBLN5', 'SEMA3G', 'BMX'],
    'Capillary_Aerocyte': ['HPGD', 'APP', 'EDNRB', 'EMCN', 'ESAM', 'STXBP6', 'CLDN5', 'CYP3A5', 'SOSTDC1', 'TBX2'],
    'Capillary_Intermediate': ['EPAS1', 'SH3BP5', 'RALA', 'PECAM1', 'WWTR1', 'GALNT18', 'ACVRL1', 'SRSF11', 'VAT1', 'IPO7'],
    'Lipofibroblast': ['WT1', 'STEAP1', 'MEDAG', 'OSR1', 'CPXM1', 'FGF7', 'CCDC71L', 'HAS1', 'C3', 'C7'],
    'Mesothelial': ['GALNT9', 'ITLN1', 'WT1', 'PRG4', 'MGARP', 'C3', 'ALDH1A3', 'DMKN', 'SLPI', 'DSC3'],
    'Platelet_Megakaryocyte': ['TUBB1', 'CMTM5', 'ACRBP', 'PF4', 'GP9', 'TMEM40', 'ITGA2B', 'GRAP2', 'GP6', 'CABP5'],
    'Natural_Killer_T': ['GNLY', 'CD7', 'CTSW', 'CCL5', 'CD2', 'ZFP36L2', 'CXCR4', 'LGALS1', 'CEBPB', 'S100A6'],
    'Fibromyocyte': ['SCX', 'FAM150A', 'HSPB3', 'RAMP1', 'CSRP1', 'CNN1', 'CST5', 'IGF1', 'CLU', 'MET'],
    'Goblet': ['S100P', 'FAM3D', 'CXCL17', 'GABRP', 'WFDC2', 'PSCA', 'AGR2', 'TXN', 'CLIC1', 'COX5A'],
    'Ionocyte': ['PHGR1', 'ASCL3', 'HEPACAM2', 'FOXI1', 'TMPRSS11E', 'ATP6V1B1', 'BSND', 'TFF3', 'STAC2', 'MFSD6L'],
    'Neuroendocrine': ['PCSK1N', 'BEX1', 'SEC11C', 'CPE', 'NENF', 'PRDX2', 'CHGA', 'EGR1', 'FAM46A', 'SCGN'],
    'Serous': ['ZG16B', 'LYZ', 'PRB4', 'PRB3', 'PRH2', 'PRR4', 'TCN1', 'LPO', 'PHB', 'TESC']
}

markers_PBMC_l3 = {
    'ASDC_mDC': ['AXL', 'LILRA4', 'SCN9A', 'CLEC4C', 'LTK', 'PPP1R14A', 'LGMN', 'SCT', 'IL3RA', 'GAS6'],
    'ASDC_pDC': ['LILRA4', 'CLEC4C', 'SCT', 'EPHB1', 'AXL', 'PROC', 'LRRC26', 'SCN9A', 'LTK', 'DNASE1L3'],
    'B_intermediate_kappa': ['MS4A1', 'IGKC', 'IGHM', 'LINC01857', 'MARCKS', 'IGHD', 'TNFRSF13B', 'CD24', 'FCRL2', 'BANK1'],
    'B_intermediate_lambda': ['MS4A1', 'IGLC2', 'IGHM', 'CD79A', 'IGLC3', 'IGHD', 'BANK1', 'TNFRSF13C', 'CD22', 'TNFRSF13B'],
    'B_memory_kappa': ['BANK1', 'IGKC', 'LINC01781', 'MS4A1', 'SSPN', 'CD79A', 'RALGPS2', 'TNFRSF13C', 'LINC00926'],
    'B_memory_lambda': ['BANK1', 'IGLC2', 'MS4A1', 'IGLC3', 'COCH', 'TNFRSF13C', 'IGHA2', 'BLK', 'TNFRSF13B', 'LINC01781'],
    'B_naive_kappa': ['IGHM', 'TCL1A', 'IGHD', 'IGHG3', 'CD79A', 'IL4R', 'CD37', 'MS4A1', 'IGKC'],
    'B_naive_lambda': ['IGHM', 'IGLC2', 'IGHD', 'IGLC3', 'CD79A', 'CXCR4', 'MS4A1', 'IL4R', 'TCL1A', 'CD79B'],
    'CD14_Mono': ['S100A9', 'CTSS', 'LYZ', 'CTSD', 'S100A8', 'VCAN', 'CD14', 'FCN1', 'S100A12', 'MS4A6A'],
    'CD16_Mono': ['LST1', 'YBX1', 'AIF1', 'FCGR3A', 'NAP1L1', 'MS4A7', 'FCER1G', 'TCF7L2', 'COTL1', 'CDKN1C'],
    'CD8_Naive': ['GZMH', 'CD4', 'GNLY', 'FGFBP2', 'IL7R', 'S100A4', 'GZMA', 'CST7', 'IL32', 'CCL5'],
    'CD4_Naive': ['TCF7', 'CD4', 'NUCB2', 'LDHB', 'TRAT1', 'SARAF', 'FHIT', 'LEF1', 'CCR7', 'IL7R'],
    'CD4_Proliferating': ['MKI67', 'TYMS', 'PCLAF', 'TOP2A', 'CENPF', 'NUSAP1', 'CENPM', 'BIRC5', 'ZWINT', 'TPX2'],
    'CD4_TCM_1': ['LTB', 'CD4', 'FYB1', 'IL7R', 'LIMS1', 'MAL', 'TMSB4X', 'TSHZ2', 'AP3M2', 'TRAC'],
    'CD4_TCM_2': ['CTLA4', 'MIR4435-2HG', 'TMSB4X', 'CD28', 'CDCA7', 'TMSB10', 'MAF', 'ITM2A', 'TRAC', 'CD27'],
    'CD4_TCM_3': ['IL7R', 'ITGB1', 'LTB', 'S100A4', 'AQP3', 'TNFRSF4', 'IL32', 'TOB1', 'PDE4D', 'HOPX'],
    'CD4_TEM_1': ['GZMK', 'IL7R', 'IL32', 'ITGB1', 'CCL5', 'GZMA', 'B2M', 'DUSP2', 'KLRB1', 'SYNE2'],
    'CD4_TEM_2': ['GZMK', 'CD4', 'TIGIT', 'IFNG-AS1', 'CD40LG', 'MALAT1', 'CD3G', 'CD3D', 'TRAC', 'CD3E'],
    'CD4_TEM_3': ['IL7R', 'CCL5', 'NOSIP', 'KLRB1', 'SERINC5', 'AQP3', 'ITGA4', 'IL32', 'TRAC', 'LTB'],
    'CD4_TEM_4': ['CCR9', 'KDF1', 'DPP4', 'CD244', 'SLC4A4', 'KLRB1', 'TMIGD2', 'CD40LG', 'IL7R', 'ODF2L'],
    'CD8_Naive': ['CD8B', 'LDHB', 'LEF1', 'LINC02446', 'CD8A', 'S100B', 'ID2', 'TCF7', 'VIM', 'CCR7'],
    'CD8_Naive_2': ['CD8B', 'CCR5', 'CHI3L2', 'SOX4', 'CD8A', 'TNFRSF9', 'CD38', 'SIRPG', 'LRRN3', 'LEF1'],
    'CD8_Proliferating': ['PCLAF', 'CD8B', 'TYMS', 'CD3D', 'CLSPN', 'CD3G', 'MKI67', 'TRAC', 'CHEK1', 'TK1'],
    'CD8_TCM_1': ['CD8B', 'SELL', 'CD8A', 'LYAR', 'ITGB1', 'NELL2', 'DUSP2', 'IL7R', 'CCL5', 'LINC01871'],
    'CD8_TCM_2': ['CD8B', 'C1orf162', 'IL7R', 'GATA3', 'YBX3', 'KRT1', 'CD8A', 'CTSW', 'INPP4B', 'LTB'],
    'CD8_TCM_3': ['CD8B', 'KLRB1', 'CD8A', 'HOPX', 'IL7R', 'KLRD1', 'CCL5', 'SCML4', 'LINC02446', 'TRAC'],
    'CD8_TEM_1': ['GZMK', 'CD8B', 'CD8A', 'CCL5', 'NKG7', 'DUSP2', 'CST7', 'IL32'],
    'CD8_TEM_2': ['CD8A', 'CMC1', 'CD8B', 'CD160', 'GZMH', 'CST7', 'KLRD1', 'CCL5', 'TIGIT', 'KLRG1'],
    'CD8_TEM_3': ['GZMK', 'CD8A', 'ITGB1', 'CD8B', 'HOPX', 'CCL5', 'KLRD1', 'NKG7', 'GNLY', 'YBX3'],
    'CD8_TEM_4': ['GZMH', 'THEMIS', 'GNLY', 'CD8A', 'ITGB1', 'FGFBP2', 'CD2', 'GZMB', 'KLRD1', 'CD8B'],
    'CD8_TEM_5': ['GZMH', 'GNLY', 'ZNF683', 'TRAC', 'KLRC2', 'TYROBP', 'CD8B', 'CD8A', 'TRGC2', 'GZMB'],
    'CD8_TEM_6': ['TRGC2', 'CD8A', 'IFNG-AS1', 'CD8B', 'ZNF683', 'KLRC2', 'NCR3', 'IKZF2', 'DUSP2', 'RTKN2'],
    'cDC1': ['WDFY4', 'C1orf54', 'CLEC9A', 'BATF3', 'CLNK', 'TSPAN33', 'FLT3', 'CADM1', 'IDO1', 'DNASE1L3'],
    'cDC2_1': ['FCER1A', 'CD14', 'CLEC10A', 'CTSS', 'ENHO', 'CD1C', 'MRC1', 'FCGR2B', 'PID1', 'IL13RA1'],
    'cDC2_2': ['FCER1A', 'BASP1', 'CD1C', 'CD74', 'CLEC10A', 'HLA-DPA1', 'ENHO', 'HLA-DPB1', 'PLD4', 'HLA-DQA1'],
    'dnT_1': ['GZMK', 'NUCB2', 'CD8B', 'GPR183', 'TCF7', 'LYAR', 'MALAT1', 'C12orf57', 'LEF1', 'LDHB'],
    'dnT_2': ['AC004585.1', 'GPR183', 'FXYD2', 'NUCB2', 'CAV1', 'CD27', 'MYB', 'TMSB4X', 'GZMK', 'FGFR1'],
    'Eryth': ['HBM', 'ALAS2', 'HBD', 'AHSP', 'SLC4A1', 'TRIM58', 'SELENBP1', 'CA1', 'IFIT1B', 'SNCA'],
    'gdT_1': ['TRDC', 'TRGC1', 'TRGV9', 'TRDV2', 'KLRD1', 'IL7R', 'KLRC1', 'DUSP2', 'GNLY', 'KLRG1'],
    'gdT_2': ['KLRC2', 'CD3G', 'KIR3DL2', 'CD3D', 'TRDC', 'TRDV1', 'ZNF683', 'KLRC1', 'TRGC1', 'GZMH'],
    'gdT_3': ['RTKN2', 'TRDC', 'TRGC2', 'LEF1', 'IKZF2', 'SOX4', 'ZNF331', 'ARID5B', 'NUCB2', 'CRTAM'],
    'gdT_4': ['TRDC', 'TIGIT', 'KLRC2', 'TRGC2', 'IKZF2', 'GCSAM', 'FCRL6', 'TRDV1', 'CST7', 'CMC1'],
    'HSPC': ['CDK6', 'SOX4', 'PRSS57', 'AC084033.3', 'ANKRD28', 'FAM30A', 'MYB', 'EGFL7', 'SPINK2', 'SMIM24'],
    'ILC': ['KIT', 'TRDC', 'IL1R1', 'SOX4', 'TNFRSF18', 'TYROBP', 'TNFRSF4', 'FCER1G', 'IL2RA', 'GATA3'],
    'MAIT': ['KLRB1', 'NKG7', 'GZMK', 'SLC4A10', 'NCR3', 'CTSW', 'IL7R', 'KLRG1', 'CEBPD', 'DUSP2'],
    'NK_Proliferating': ['STMN1', 'KLRF1', 'TYMS', 'FCER1G', 'PCNA', 'TYROBP', 'CLSPN', 'TRDC', 'PCLAF', 'SMC2'],
    'NK_1': ['FGFBP2', 'KLRC2', 'GNLY', 'S100A4', 'CD3E', 'CST7', 'LGALS1', 'PRF1', 'NKG7', 'GZMB'],
    'NK_2': ['NKG7', 'FCER1G', 'PRF1', 'KLRB1', 'SPON2', 'GZMB', 'FGFBP2', 'IGFBP7', 'CST7', 'B2M'],
    'NK_3': ['KLRF1', 'CCL5', 'TRDC', 'SYNE2', 'KLRC1', 'CMC1', 'XCL2', 'KLRB1', 'KLRD1', 'IL2RB'],
    'NK_4': ['XCL2', 'SELL', 'XCL1', 'GZMK', 'KLRC1', 'SPTSSB', 'KLRF1', 'IL2RB', 'TCF7', 'TRDC'],
    'NK_CD56bright': ['XCL2', 'GPR183', 'SELL', 'IL2RB', 'CD44', 'GZMK', 'KLRF1', 'TPT1', 'KLRC1', 'XCL1'],
    'pDC': ['CCDC50', 'UGCG', 'TCF4', 'LILRA4', 'IRF8', 'IL3RA', 'PLD4', 'IRF7', 'SERPINF1', 'ITM2C'],
    'Plasma': ['MZB1', 'JCHAIN', 'TNFRSF17', 'ITM2C', 'DERL3', 'TXNDC5', 'POU2AF1', 'IGHA1', 'TXNDC11', 'CD79A'],
    'Plasmablast': ['TYMS', 'TNFRSF17', 'SHCBP1', 'TK1', 'KNL1', 'ASPM', 'TXNDC5', 'TPX2', 'RRM2', 'BIRC5'],
    'Platelet': ['GNG11', 'PPBP', 'NRGN', 'PF4', 'CAVIN2', 'TUBB1', 'HIST1H2AC', 'PRKAR2B', 'CLU', 'F13A1'],
    'Treg_Memory': ['RTKN2', 'B2M', 'TIGIT', 'FCRL3', 'S100A4', 'AC133644.2', 'CTLA4', 'FOXP3', 'IKZF2', 'TRAC'],
    'Treg_Naive': ['RTKN2', 'LEF1', 'FOXP3', 'C12orf57', 'IL2RA', 'TOMM7', 'CCR7', 'TRAC', 'CD4', 'LDHB']
}


markers_PBMC_l3_up = {
    'ASDC_mDC': ['AXL', 'LILRA4', 'SCN9A', 'CLEC4C', 'LTK', 'PPP1R14A', 'LGMN', 'SCT', 'IL3RA', 'GAS6'],
    'ASDC_pDC': ['LILRA4', 'CLEC4C', 'SCT', 'EPHB1', 'AXL', 'PROC', 'LRRC26', 'SCN9A', 'LTK', 'DNASE1L3'],
    'B_intermediate_kappa': ['MS4A1', 'IGKC', 'IGHM', 'LINC01857', 'MARCKS', 'IGHD', 'TNFRSF13B', 'CD24', 'FCRL2', 'BANK1'],
    'B_intermediate_lambda': ['MS4A1', 'IGLC2', 'IGHM', 'CD79A', 'IGLC3', 'IGHD', 'BANK1', 'TNFRSF13C', 'CD22', 'TNFRSF13B'],
    'B_memory_kappa': ['BANK1', 'IGKC', 'LINC01781', 'MS4A1', 'SSPN', 'CD79A', 'RALGPS2', 'TNFRSF13C', 'LINC00926'],
    'B_memory_lambda': ['BANK1', 'IGLC2', 'MS4A1', 'IGLC3', 'COCH', 'TNFRSF13C', 'IGHA2', 'BLK', 'TNFRSF13B', 'LINC01781'],
    'B_naive_kappa': ['IGHM', 'TCL1A', 'IGHD', 'IGHG3', 'CD79A', 'IL4R', 'CD37', 'MS4A1', 'IGKC'],
    'B_naive_lambda': ['IGHM', 'IGLC2', 'IGHD', 'IGLC3', 'CD79A', 'CXCR4', 'MS4A1', 'IL4R', 'TCL1A', 'CD79B'],
    'CD14_Mono': ['S100A9', 'CTSS', 'LYZ', 'CTSD', 'S100A8', 'VCAN', 'CD14', 'FCN1', 'S100A12', 'MS4A6A'],
    'CD16_Mono': ['LST1', 'YBX1', 'AIF1', 'FCGR3A', 'NAP1L1', 'MS4A7', 'FCER1G', 'TCF7L2', 'COTL1', 'CDKN1C'],
    'CD8_Naive': ['GZMH', 'CD4', 'GNLY', 'FGFBP2', 'IL7R', 'S100A4', 'GZMA', 'CST7', 'IL32', 'CCL5'],
    'CD4_Naive': ['TCF7', 'CD4', 'NUCB2', 'LDHB', 'TRAT1', 'SARAF', 'FHIT', 'LEF1', 'CCR7', 'IL7R'],
    'CD4_Proliferating': ['MKI67', 'TYMS', 'PCLAF', 'TOP2A', 'CENPF', 'NUSAP1', 'CENPM', 'BIRC5', 'ZWINT', 'TPX2'],
    'CD4_TCM_1': ['LTB', 'CD4', 'FYB1', 'IL7R', 'LIMS1', 'MAL', 'TMSB4X', 'TSHZ2', 'AP3M2', 'TRAC'],
    'CD4_TCM_2': ['CTLA4', 'MIR4435-2HG', 'TMSB4X', 'CD28', 'CDCA7', 'TMSB10', 'MAF', 'ITM2A', 'TRAC', 'CD27'],
    'CD4_TCM_3': ['IL7R', 'ITGB1', 'LTB', 'S100A4', 'AQP3', 'TNFRSF4', 'IL32', 'TOB1', 'PDE4D', 'HOPX'],
    'CD4_TEM_1': ['GZMK', 'IL7R', 'IL32', 'ITGB1', 'CCL5', 'GZMA', 'B2M', 'DUSP2', 'KLRB1', 'SYNE2'],
    'CD4_TEM_2': ['GZMK', 'CD4', 'TIGIT', 'IFNG-AS1', 'CD40LG', 'MALAT1', 'CD3G', 'CD3D', 'TRAC', 'CD3E'],
    'CD4_TEM_3': ['IL7R', 'CCL5', 'NOSIP', 'KLRB1', 'SERINC5', 'AQP3', 'ITGA4', 'IL32', 'TRAC', 'LTB'],
    'CD4_TEM_4': ['CCR9', 'KDF1', 'DPP4', 'CD244', 'SLC4A4', 'KLRB1', 'TMIGD2', 'CD40LG', 'IL7R', 'ODF2L'],
    'CD8_Naive': ['CD8B', 'LDHB', 'LEF1', 'LINC02446', 'CD8A', 'S100B', 'ID2', 'TCF7', 'VIM', 'CCR7'],
    'CD8_Naive_2': ['CD8B', 'CCR5', 'CHI3L2', 'SOX4', 'CD8A', 'TNFRSF9', 'CD38', 'SIRPG', 'LRRN3', 'LEF1'],
    'CD8_Proliferating': ['PCLAF', 'CD8B', 'TYMS', 'CD3D', 'CLSPN', 'CD3G', 'MKI67', 'TRAC', 'CHEK1', 'TK1'],
    'CD8_TCM_1': ['CD8B', 'SELL', 'CD8A', 'LYAR', 'ITGB1', 'NELL2', 'DUSP2', 'IL7R', 'CCL5', 'LINC01871'],
    'CD8_TCM_2': ['CD8B', 'C1orf162', 'IL7R', 'GATA3', 'YBX3', 'KRT1', 'CD8A', 'CTSW', 'INPP4B', 'LTB'],
    'CD8_TCM_3': ['CD8B', 'KLRB1', 'CD8A', 'HOPX', 'IL7R', 'KLRD1', 'CCL5', 'SCML4', 'LINC02446', 'TRAC'],
    'CD8_TEM_1': ['GZMK', 'CD8B', 'CD8A', 'CCL5', 'NKG7', 'DUSP2', 'CST7', 'IL32'],
    'CD8_TEM_2': ['CD8A', 'CMC1', 'CD8B', 'CD160', 'GZMH', 'CST7', 'KLRD1', 'CCL5', 'TIGIT', 'KLRG1'],
    'CD8_TEM_3': ['GZMK', 'CD8A', 'ITGB1', 'CD8B', 'HOPX', 'CCL5', 'KLRD1', 'NKG7', 'GNLY', 'YBX3'],
    'CD8_TEM_4': ['GZMH', 'THEMIS', 'GNLY', 'CD8A', 'ITGB1', 'FGFBP2', 'CD2', 'GZMB', 'KLRD1', 'CD8B'],
    'CD8_TEM_5': ['GZMH', 'GNLY', 'ZNF683', 'TRAC', 'KLRC2', 'TYROBP', 'CD8B', 'CD8A', 'TRGC2', 'GZMB'],
    'CD8_TEM_6': ['TRGC2', 'CD8A', 'IFNG-AS1', 'CD8B', 'ZNF683', 'KLRC2', 'NCR3', 'IKZF2', 'DUSP2', 'RTKN2'],
    'NK': ['WDFY4', 'C1orf54', 'CLEC9A', 'BATF3', 'CLNK', 'TSPAN33', 'FLT3', 'CADM1', 'IDO1', 'DNASE1L3'],
    'cDC2_1': ['FCER1A', 'CD14', 'CLEC10A', 'CTSS', 'ENHO', 'CD1C', 'MRC1', 'FCGR2B', 'PID1', 'IL13RA1'],
    'cDC2_2': ['FCER1A', 'BASP1', 'CD1C', 'CD74', 'CLEC10A', 'HLA-DPA1', 'ENHO', 'HLA-DPB1', 'PLD4', 'HLA-DQA1'],
    'dnT_1': ['GZMK', 'NUCB2', 'CD8B', 'GPR183', 'TCF7', 'LYAR', 'MALAT1', 'C12orf57', 'LEF1', 'LDHB'],
    'dnT_2': ['AC004585.1', 'GPR183', 'FXYD2', 'NUCB2', 'CAV1', 'CD27', 'MYB', 'TMSB4X', 'GZMK', 'FGFR1'],
    'Eryth': ['HBM', 'ALAS2', 'HBD', 'AHSP', 'SLC4A1', 'TRIM58', 'SELENBP1', 'CA1', 'IFIT1B', 'SNCA'],
    'gdT_1': ['TRDC', 'TRGC1', 'TRGV9', 'TRDV2', 'KLRD1', 'IL7R', 'KLRC1', 'DUSP2', 'GNLY', 'KLRG1'],
    'gdT_2': ['KLRC2', 'CD3G', 'KIR3DL2', 'CD3D', 'TRDC', 'TRDV1', 'ZNF683', 'KLRC1', 'TRGC1', 'GZMH'],
    'gdT_3': ['RTKN2', 'TRDC', 'TRGC2', 'LEF1', 'IKZF2', 'SOX4', 'ZNF331', 'ARID5B', 'NUCB2', 'CRTAM'],
    'gdT_4': ['TRDC', 'TIGIT', 'KLRC2', 'TRGC2', 'IKZF2', 'GCSAM', 'FCRL6', 'TRDV1', 'CST7', 'CMC1'],
    'HSPC': ['CDK6', 'SOX4', 'PRSS57', 'AC084033.3', 'ANKRD28', 'FAM30A', 'MYB', 'EGFL7', 'SPINK2', 'SMIM24'],
    'DC': ['KIT', 'TRDC', 'IL1R1', 'SOX4', 'TNFRSF18', 'TYROBP', 'TNFRSF4', 'FCER1G', 'IL2RA', 'GATA3'],
    'MAIT': ['KLRB1', 'NKG7', 'GZMK', 'SLC4A10', 'NCR3', 'CTSW', 'IL7R', 'KLRG1', 'CEBPD', 'DUSP2'],
    'NK_Proliferating': ['STMN1', 'KLRF1', 'TYMS', 'FCER1G', 'PCNA', 'TYROBP', 'CLSPN', 'TRDC', 'PCLAF', 'SMC2'],
    'NK_1': ['GNLY', 'KLRC2', 'CD3E', 'CST7', 'PRF1', 'GZMB', 'LGALS1', 'FGFBP2', 'S100A4', 'NKG7'],
    'NK_2': ['IGFBP7', 'SPON2', 'CST7', 'B2M', 'PRF1', 'GZMB', 'KLRB1', 'FCER1G', 'FGFBP2', 'NKG7'],
    'NK_3': ['KLRC1', 'IL2RB', 'TRDC', 'CMC1', 'KLRB1', 'KLRF1', 'SYNE2', 'XCL2', 'CCL5', 'KLRD1'],
    'NK_CD56bright': ['XCL2', 'GPR183', 'SELL', 'IL2RB', 'CD44', 'GZMK', 'KLRF1', 'TPT1', 'KLRC1', 'XCL1'],
    'pDC': ['CCDC50', 'UGCG', 'TCF4', 'LILRA4', 'IRF8', 'IL3RA', 'PLD4', 'IRF7', 'SERPINF1', 'ITM2C'],
    'Plasma': ['MZB1', 'JCHAIN', 'TNFRSF17', 'ITM2C', 'DERL3', 'TXNDC5', 'POU2AF1', 'IGHA1', 'TXNDC11', 'CD79A'],
    'Plasmablast': ['TYMS', 'TNFRSF17', 'SHCBP1', 'TK1', 'KNL1', 'ASPM', 'TXNDC5', 'TPX2', 'RRM2', 'BIRC5'],
    'Platelet': ['GNG11', 'PPBP', 'NRGN', 'PF4', 'CAVIN2', 'TUBB1', 'HIST1H2AC', 'PRKAR2B', 'CLU', 'F13A1'],
    'Treg_Memory': ['RTKN2', 'B2M', 'TIGIT', 'FCRL3', 'S100A4', 'AC133644.2', 'CTLA4', 'FOXP3', 'IKZF2', 'TRAC'],
    'Treg_Naive': ['RTKN2', 'LEF1', 'FOXP3', 'C12orf57', 'IL2RA', 'TOMM7', 'CCR7', 'TRAC', 'CD4', 'LDHB']
}





markers_PBMC_l2 = {
    'B_intermediate': ['MS4A1', 'TNFRSF13B', 'IGHM', 'IGHD', 'AIM2', 'CD79A', 'LINC01857', 'RALGPS2', 'BANK1', 'CD79B'],
    'B_memory': ['MS4A1', 'COCH', 'AIM2', 'BANK1', 'SSPN', 'CD79A', 'TEX9', 'RALGPS2', 'TNFRSF13C', 'LINC01781'],
    'B_naive': ['IGHM', 'IGHD', 'CD79A', 'IL4R', 'MS4A1', 'CXCR4', 'BTG1', 'TCL1A', 'CD79B', 'YBX3'],
    'Plasmablast': ['IGHA2', 'MZB1', 'TNFRSF17', 'DERL3', 'TXNDC5', 'TNFRSF13B', 'POU2AF1', 'CPNE5', 'HRASLS2', 'NT5DC2'],
    'CD4_CTL': ['GZMH', 'CD4', 'FGFBP2', 'ITGB1', 'GZMA', 'CST7', 'GNLY', 'B2M', 'IL32', 'NKG7'],
    'CD4_Naive': ['TCF7', 'CD4', 'CCR7', 'IL7R', 'FHIT', 'LEF1', 'MAL', 'NOSIP', 'LDHB', 'PIK3IP1'],
    'CD4_Proliferating': ['MKI67', 'TOP2A', 'PCLAF', 'CENPF', 'TYMS', 'NUSAP1', 'ASPM', 'PTTG1', 'TPX2', 'RRM2'],
    'CD4_TCM': ['IL7R', 'TMSB10', 'CD4', 'ITGB1', 'LTB', 'TRAC', 'AQP3', 'LDHB', 'IL32', 'MAL'],
    'CD4_TEM': ['IL7R', 'CCL5', 'FYB1', 'GZMK', 'IL32', 'GZMA', 'KLRB1', 'TRAC', 'LTB', 'AQP3'],
    'Treg': ['RTKN2', 'FOXP3', 'AC133644.2', 'CD4', 'IL2RA', 'TIGIT', 'CTLA4', 'FCRL3', 'LAIR2', 'IKZF2'],
    'CD8_Naive': ['CD8B', 'S100B', 'CCR7', 'RGS10', 'NOSIP', 'LINC02446', 'LEF1', 'CRTAM', 'CD8A', 'OXNAD1'],
    'CD8_Proliferating': ['MKI67', 'CD8B', 'TYMS', 'TRAC', 'PCLAF', 'CD3D', 'CLSPN', 'CD3G', 'TK1', 'RRM2'],
    'CD8_TCM': ['CD8B', 'ANXA1', 'CD8A', 'KRT1', 'LINC02446', 'YBX3', 'IL7R', 'TRAC', 'NELL2', 'LDHB'],
    'CD8_TEM': ['CCL5', 'GZMH', 'CD8A', 'TRAC', 'KLRD1', 'NKG7', 'GZMK', 'CST7', 'CD8B', 'TRGC2'],
    'ASDC': ['PPP1R14A', 'LILRA4', 'AXL', 'IL3RA', 'SCT', 'SCN9A', 'LGMN', 'DNASE1L3', 'CLEC4C', 'GAS6'],
    'cDC1': ['CLEC9A', 'DNASE1L3', 'C1orf54', 'IDO1', 'CLNK', 'CADM1', 'FLT3', 'ENPP1', 'XCR1', 'NDRG2'],
    'cDC2': ['FCER1A', 'HLA-DQA1', 'CLEC10A', 'CD1C', 'ENHO', 'PLD4', 'GSN', 'SLC38A1', 'NDRG2', 'AFF3'],
    'pDC': ['ITM2C', 'PLD4', 'SERPINF1', 'LILRA4', 'IL3RA', 'TPM2', 'MZB1', 'SPIB', 'IRF4', 'SMPD3'],
    'CD14_Mono': ['S100A9', 'CTSS', 'S100A8', 'LYZ', 'VCAN', 'S100A12', 'IL1B', 'CD14', 'G0S2', 'FCN1'],
    'CD16_Mono': ['CDKN1C', 'FCGR3A', 'PTPRC', 'LST1', 'IER5', 'MS4A7', 'RHOC', 'IFITM3', 'AIF1', 'HES4'],
    'NK': ['GNLY', 'TYROBP', 'NKG7', 'FCER1G', 'GZMB', 'TRDC', 'PRF1', 'FGFBP2', 'SPON2', 'KLRF1'],
    'NK_Proliferating': ['MKI67', 'KLRF1', 'TYMS', 'TRDC', 'TOP2A', 'FCER1G', 'PCLAF', 'CD247', 'CLSPN', 'ASPM'],
    'NK_CD56bright': ['XCL2', 'FCER1G', 'SPINK2', 'TRDC', 'KLRC1', 'XCL1', 'SPTSSB', 'PPP1R9A', 'NCAM1', 'TNFRSF11A'],
    'Eryth': ['HBD', 'HBM', 'AHSP', 'ALAS2', 'CA1', 'SLC4A1', 'IFIT1B', 'TRIM58', 'SELENBP1', 'TMCC2'],
    'HSPC': ['SPINK2', 'PRSS57', 'CYTL1', 'EGFL7', 'GATA2', 'CD34', 'SMIM24', 'AVP', 'MYB', 'LAPTM4B'],
    'ILC': ['KIT', 'TRDC', 'TTLL10', 'LINC01229', 'SOX4', 'KLRB1', 'TNFRSF18', 'TNFRSF4', 'IL1R1', 'HPGDS'],
    'Platelet': ['PPBP', 'PF4', 'NRGN', 'GNG11', 'CAVIN2', 'TUBB1', 'CLU', 'HIST1H2AC', 'RGS18', 'GP9'],
    'dnT': ['PTPN3', 'MIR4422HG', 'NUCB2', 'CAV1', 'DTHD1', 'GZMA', 'MYB', 'FXYD2', 'GZMK', 'AC004585.1'],
    'gdT': ['TRDC', 'TRGC1', 'TRGC2', 'KLRC1', 'NKG7', 'TRDV2', 'CD7', 'TRGV9', 'KLRD1', 'KLRG1'],
    'MAIT': ['KLRB1', 'NKG7', 'GZMK', 'IL7R', 'SLC4A10', 'GZMA', 'CXCR6', 'PRSS35', 'RBM24', 'NCR3']
}


markers_lung_v2 = {
    'Alveolar macrophages': ['MS4A7', 'C1QA', 'HLA-DQB1', 'HLA-DMA', 'HLA-DPB1', 'HLA-DPA1', 'ACP5', 'C1QC', 'CTSS', 'HLA-DQA1'],
    'NK cells': ['GZMA', 'CD7', 'CCL4', 'CST7', 'NKG7', 'GNLY', 'CTSW', 'CCL5', 'GZMB', 'PRF1'],
    'AT2': ['SEPP1', 'PGC', 'NAPSA', 'SFTPD', 'SLC34A2', 'CYB5A', 'MUC1', 'S100A14', 'SFTA2', 'SFTA3'],
    'Alveolar Mφ CCL3+': ['MCEMP1', 'UPP1', 'HLA-DQA1', 'C5AR1', 'HLA-DMA', 'AIF1', 'LST1', 'LINC01272', 'MRC1', 'CCL18'],
    'Suprabasal': ['PRDX2', 'KRT19', 'SFN', 'TACSTD2', 'KRT5', 'LDHB', 'KRT17', 'KLK11', 'S100A2', 'SERPINB4'],
    'Basal resting': ['CYR61', 'PERP', 'IGFBP2', 'KRT19', 'KRT5', 'KRT17', 'KRT15', 'S100A2', 'LAMB3', 'BCAM'],
    'EC venous pulmonary': ['VWF', 'MGP', 'GNG11', 'RAMP2', 'SPARCL1', 'IGFBP7', 'IFI27', 'CLDN5', 'ACKR1', 'AQP1'],
    'CD8 T cells': ['CD8A', 'CD3E', 'CCL4', 'CD2', 'CXCR4', 'GZMA', 'NKG7', 'IL32', 'CD3D', 'CCL5'],
    'EC arterial': ['SPARCL1', 'SOX17', 'IFI27', 'TM4SF1', 'A2M', 'CLEC14A', 'GIMAP7', 'CRIP2', 'CLDN5', 'PECAM1'],
    'Peribronchial fibroblasts': ['IGFBP7', 'COL1A2', 'COL3A1', 'A2M', 'BGN', 'DCN', 'MGP', 'LUM', 'MFAP4', 'C1S'],
    'CD4 T cells': ['CORO1A', 'KLRB1', 'CD3E', 'LTB', 'CXCR4', 'IL7R', 'TRAC', 'IL32', 'CD2', 'CD3D'],
    'AT1': ['SFTA2', 'CEACAM6', 'FXYD3', 'CAV1', 'TSPAN13', 'KRT7', 'ADIRF', 'HOPX', 'AGER', 'EMP2'],
    'Multiciliated (non-nasal)': ['SNTN', 'FAM229B', 'TMEM231', 'C5orf49', 'C12orf75', 'GSTA1', 'C11orf97', 'RP11-356K23.1', 'CD24', 'RP11-295M3.4'],
    'Plasma cells': ['ITM2C', 'TNFRSF17', 'FKBP11', 'IGKC', 'IGHA1', 'IGHG1', 'CD79A', 'JCHAIN', 'MZB1', 'ISG20'],
    'Goblet (nasal)': ['KRT7', 'MUC1', 'MUC5AC', 'MSMB', 'CP', 'LMO7', 'LCN2', 'CEACAM6', 'BPIFB1', 'PIGR'],
    'Club (nasal)': ['ELF3', 'C19orf33', 'KRT8', 'KRT19', 'TACSTD2', 'MUC1', 'S100A14', 'CXCL17', 'PSCA', 'FAM3D'],
    'SM activated stress response': ['C11orf96', 'HES4', 'PLAC9', 'FLNA', 'KANK2', 'TPM2', 'PLN', 'SELM', 'GPX3', 'LBH'],
    'Classical monocytes': ['LST1', 'IL1B', 'LYZ', 'COTL1', 'S100A9', 'VCAN', 'S100A8', 'S100A12', 'AIF1', 'FCN1'],
    'Monocyte derived Mφ': ['LYZ', 'ACP5', 'TYROBP', 'LGALS1', 'CD68', 'AIF1', 'CTSL', 'EMP3', 'FCER1G', 'LAPTM5'],
    'Alveolar Mφ proliferating': ['H2AFV', 'STMN1', 'LSM4', 'GYPC', 'PTTG1', 'KIAA0101', 'FABP4', 'CKS1B', 'UBE2C', 'HMGN2'],
    'Club (non-nasal)': ['SCGB3A1', 'CYP2F1', 'GSTA1', 'HES4', 'TSPAN8', 'TFF3', 'MSMB', 'BPIFB1', 'SCGB1A1', 'PIGR'],
    'SMG serous (bronchial)': ['AZGP1', 'ZG16B', 'PIGR', 'NDRG2', 'LPO', 'C6orf58', 'DMBT1', 'PRB3', 'FAM3D', 'RP11-1143G9.4'],
    'EC venous systemic': ['VWF', 'MGP', 'GNG11', 'PLVAP', 'RAMP2', 'SPARCL1', 'IGFBP7', 'A2M', 'CLEC14A', 'ACKR1'],
    'Non classical monocytes': ['PSAP', 'FCGR3A', 'FCN1', 'CORO1A', 'COTL1', 'FCER1G', 'LAPTM5', 'CTSS', 'AIF1', 'LST1'],
    'EC general capillary': ['EPAS1', 'GNG11', 'IFI27', 'TM4SF1', 'EGFL7', 'AQP1', 'VWF', 'FCN3', 'SPARCL1', 'CLDN5'],
    'Adventitial fibroblasts': ['COL6A2', 'SFRP2', 'IGFBP7', 'IGFBP6', 'COL3A1', 'C1S', 'MMP2', 'MGP', 'SPARC', 'COL1A2'],
    'Lymphatic EC mature': ['PPFIBP1', 'GNG11', 'RAMP2', 'CCL21', 'MMRN1', 'IGFBP7', 'SDPR', 'TM4SF1', 'CLDN5', 'ECSCR'],
    'EC aerocyte capillary': ['EMCN', 'HPGD', 'IFI27', 'CA4', 'EGFL7', 'AQP1', 'IL1RL1', 'SPARCL1', 'SDPR', 'CLDN5'],
    'Smooth muscle': ['PRKCDBP', 'NDUFA4L2', 'MYL9', 'ACTA2', 'MGP', 'CALD1', 'TPM1', 'TAGLN', 'IGFBP7', 'TPM2'],
    'Alveolar fibroblasts': ['LUM', 'COL6A1', 'CYR61', 'C1R', 'COL1A2', 'MFAP4', 'A2M', 'C1S', 'ADH1B', 'GPX3'],
    'Multiciliated (nasal)': ['RP11-356K23.1', 'EFHC1', 'CAPS', 'ROPN1L', 'RSPH1', 'C9orf116', 'TMEM190', 'DNALI1', 'PIFO', 'ODF3B'],
    'Goblet (bronchial)': ['MUC5AC', 'MSMB', 'PI3', 'MDK', 'ANKRD36C', 'TFF3', 'PIGR', 'SAA1', 'CP', 'BPIFB1'],
    'Neuroendocrine': ['UCHL1', 'TFF3', 'APOA1BP', 'CLDN3', 'SEC11C', 'NGFRAP1', 'SCG5', 'HIGD1A', 'PHGR1', 'CD24'],
    'Lymphatic EC differentiating': ['AKAP12', 'TFF3', 'SDPR', 'CLDN5', 'TCF4', 'TFPI', 'TIMP3', 'GNG11', 'CCL21', 'IGFBP7'],
    'DC2': ['ITGB2', 'LAPTM5', 'HLA-DRB1', 'HLA-DPB1', 'HLA-DPA1', 'HLA-DMB', 'HLA-DQB1', 'HLA-DQA1', 'HLA-DMA', 'LST1'],
    'Transitional Club AT2': ['CXCL17', 'C16orf89', 'RNASE1', 'KRT7', 'SCGB1A1', 'PIGR', 'SCGB3A2', 'KLK11', 'SFTA1P', 'FOLR1'],
    'DC1': ['HLA-DPA1', 'CPNE3', 'CORO1A', 'CPVL', 'C1orf54', 'WDFY4', 'LSP1', 'HLA-DQB1', 'HLA-DQA1', 'HLA-DMA'],
    'Myofibroblasts': ['CALD1', 'CYR61', 'TAGLN', 'MT1X', 'PRELP', 'TPM2', 'GPX3', 'CTGF', 'IGFBP5', 'SPARCL1'],
    'B cells': ['CD69', 'CORO1A', 'LIMD2', 'BANK1', 'LAPTM5', 'CXCR4', 'LTB', 'CD79A', 'CD37', 'MS4A1'],
    'Mast cells': ['VWA5A', 'RGS13', 'C1orf186', 'HPGDS', 'CPA3', 'GATA2', 'MS4A2', 'KIT', 'TPSAB1', 'TPSB2'],
    'Interstitial Mφ perivascular': ['MRC1', 'RNASE1', 'FGL2', 'RNASE6', 'HLA-DPA1', 'GPR183', 'CD14', 'HLA-DPB1', 'MS4A6A', 'AIF1'],
    'SMG mucous': ['FKBP11', 'TCN1', 'GOLM1', 'TFF3', 'PIGR', 'KLK11', 'MARCKSL1', 'CRACR2B', 'SELM', 'MSMB'],
    'AT2 proliferating': ['CDK1', 'LSM3', 'CKS1B', 'EIF1AX', 'UBE2C', 'MRPL14', 'PRC1', 'CENPW', 'EMP2', 'DHFR'],
    'Goblet (subsegmental)': ['MDK', 'MUC5B', 'SCGB1A1', 'CP', 'C3', 'TSPAN8', 'TFF3', 'MSMB', 'PIGR', 'BPIFB1'],
    'Pericytes': ['MYL9', 'SPARC', 'SPARCL1', 'IGFBP7', 'COL4A1', 'GPX3', 'PDGFRB', 'CALD1', 'COX4I2', 'TPM2'],
    'SMG duct': ['PIP', 'ZG16B', 'PIGR', 'SAA1', 'MARCKSL1', 'ALDH1A3', 'SELM', 'LTF', 'RARRES1', 'AZGP1'],
    'Mesothelium': ['CEBPD', 'LINC01133', 'MRPL33', 'UPK3B', 'CFB', 'SEPP1', 'EID1', 'HP', 'CUX1', 'MRPS21'],
    'SMG serous (nasal)': ['ZG16B', 'MUC7', 'C6orf58', 'PRB3', 'LTF', 'LYZ', 'PRR4', 'AZGP1', 'PIGR', 'RP11-1143G9.4'],
    'Ionocyte': ['FOXI1', 'ATP6V1A', 'GOLM1', 'TMEM61', 'SEC11C', 'SCNN1B', 'ASCL3', 'CLCNKB', 'HEPACAM2', 'CD24'],
    'Alveolar Mφ MT-positive': ['GSTO1', 'LGALS1', 'CTSZ', 'MT2A', 'APOC1', 'CTSL', 'UPP1', 'CCL18', 'FABP4', 'MT1X'],
    'Fibromyocytes': ['NEXN', 'ACTG2', 'LMOD1', 'IGFBP7', 'PPP1R14A', 'DES', 'FLNA', 'TPM2', 'PLN', 'SELM'],
    'Deuterosomal': ['RSPH9', 'PIFO', 'RUVBL2', 'C11orf88', 'FAM183A', 'MORN2', 'SAXO2', 'CFAP126', 'FAM229B', 'C5orf49'],
    'Tuft': ['MUC20', 'KHDRBS1', 'ZNF428', 'BIK', 'CRYM', 'LRMP', 'HES6', 'KIT', 'AZGP1', 'RASSF6'],
    'Plasmacytoid DCs': ['IL3RA', 'TCF4', 'LTB', 'GZMB', 'JCHAIN', 'ITM2C', 'IRF8', 'PLD4', 'IRF7', 'C12orf75'],
    'T cells proliferating': ['TRAC', 'HMGN2', 'IL32', 'CORO1A', 'ARHGDIB', 'STMN1', 'RAC2', 'IL2RG', 'HMGB2', 'CD3D'],
    'Subpleural fibroblasts': ['SERPING1', 'C1R', 'COL1A2', 'NNMT', 'COL3A1', 'MT1E', 'MT1X', 'PLA2G2A', 'SELM', 'MT1M'],
    'Lymphatic EC proliferating': ['S100A16', 'TUBB', 'HMGN2', 'COX20', 'LSM2', 'HMGN1', 'ARPC1A', 'ECSCR', 'EID1', 'MARCKS'],
    'Migratory DCs': ['IL2RG', 'HLA-DRB5', 'TMEM176A', 'BIRC3', 'TYMP', 'CCL22', 'SYNGR2', 'CD83', 'LSP1', 'HLA-DQA1']
}


markers_heart = {
    'Adipocyte': ['GPAM', 'ACACB', 'GHR', 'PDE3B', 'FASN', 'PRKAR2B', 'MGST1', 'PLIN1', 'LINC00598', 'SLC19A3'],
    'Arterial Endothelial': ['LINC00639', 'ARL15', 'MECOM', 'IGFBP3', 'PCSK5', 'TM4SF1', 'EFNB2', 'DLL4', 'HEY1', 'GJA5'],
    'Atrial Cardiomyocyte': ['MYL7', 'FGF12', 'MYH6', 'ANKRD1', 'RYR2', 'ERBB4', 'TTN', 'CMYA5', 'NPPA', 'MYL7'],
    'B': ['IGKC', 'IGHM', 'IGHA1', 'AFF3', 'BANK1', 'IFNG-AS1', 'GNG7', 'MZB1', 'RALGPS2', 'MS4A1'],
    'Capillary Endothelial': ['BTNL9', 'FABP5', 'IFI27', 'RGCC', 'F8', 'MGLL', 'CA4', 'ADGRF5', 'ABLIM3', 'FLT1'],
    'Endocardial': ['PCDH15', 'EGFL7', 'NRG3', 'LEPR', 'NRG1', 'POSTN', 'PCDH7', 'TMEM132C', 'CDH11', 'TMEM108'],
    'Endothelial': ['MYRIP', 'POSTN', 'NRG3', 'LDB2', 'SLCO2A1', 'ANO2', 'VWF', 'FAM155A', 'IL1R1', 'KCNIP4'],
    'Fibroblast': ['DCN', 'APOD', 'ACSM3', 'CFD', 'ABCA9', 'NEGR1', 'LAMA2', 'ABCA6', 'ABCA8', 'KAZN'],
    'ILC': ['KIT', 'IL18R1', 'RD3', 'CPA3', 'SLC38A11', 'SLC8A3', 'TPSAB1', 'HDC', 'MS4A2', 'OR8A1'],
    'Lymphatic Endothelial': ['CCL21', 'MMRN1', 'RELN', 'PKHD1L1', 'SEMA3A', 'PPFIBP1', 'TFPI', 'SNTG2', 'EFNA5', 'PROX1'],
    'Macrophage': ['F13A1', 'RBPJ', 'MRC1', 'CD163', 'COLEC12', 'RBM47', 'FMN1', 'MS4A6A', 'STAB1', 'FRMD4B'],
    'Mast': ['SLC24A3', 'IL18R1', 'CPA3', 'NTM', 'KIT', 'SLC8A3', 'SLC38A11', 'CDK15', 'HPGDS', 'RAB27B'],
    'Mesothelial': ['ITLN1', 'PDZRN4', 'SLC39A8', 'PRG4', 'GFPT2', 'C3', 'KCNT2', 'WWC1', 'HAS1', 'RBFOX1'],
    'Monocyte/cDC': ['S100A8', 'S100A9', 'CD163', 'MSR1', 'SLC11A1', 'FMN1', 'RBM47', 'ITGAX', 'ADAM28', 'FLT3', 'CD74'],
    'Neuronal': ['NRXN1', 'NRXN3', 'CADM2', 'XKR4', 'CDH19', 'ADGRB3', 'CHL1', 'NCAM2', 'KIRREL3', 'SORCS1'],
    'NK': ['GNLY', 'NKG7', 'CD247', 'TXK', 'KLRD1', 'CCL4', 'CCL5', 'AOAH', 'PRF1', 'PARP8'],
    'Pericyte': ['RGS5', 'ABCC9', 'GUCY1A2', 'EGFLAM', 'DLC1', 'FRMD3', 'PDGFRB', 'AGT', 'PLA2G5', 'EPS8'],
    'Smooth Muscle': ['MYH11', 'ITGA8', 'ACTA2', 'CARMN', 'KCNAB1', 'TAGLN', 'ZFHX3', 'PRKG1', 'NTRK3', 'RCAN2'],
    'T': ['IL7R', 'THEMIS', 'ITK', 'PARP8', 'CAMK4', 'SKAP1', 'TC2N', 'BCL11B', 'PTPRC', 'PRKCQ'],
    'Venous Endothelial': ['TPO', 'ACKR1', 'TSHZ2', 'RAMP3', 'FAM155A', 'ABCB1', 'CALCRL', 'CYYR1', 'IGFBP5', 'EPHB4', 'NPP2'],
    'Ventricular Cardiomyocyte': ['FHL2', 'RYR2', 'MLIP', 'TTN', 'CTNNA3', 'RBM20', 'FGF12', 'SLC8A1', 'MYOM1', 'MYH7', 'MYL2']
}


markers_human_cortex = {
    'Astro': ['SLC1A2', 'ADGRV1', 'SLC1A3', 'GPC5', 'RNF219.AS1', 'ARHGAP24', 'CST3', 'HPSE2', 'AQP4', 'COL5A3'],
    'Endo': ['EBF1', 'ABCG2', 'CLDN5', 'FLI1', 'LEF1', 'EMCN', 'IFI27', 'HLA.E', 'ADGRL4', 'CLEC3B'],
    'L2/3 IT': ['CBLN2', 'EPHA6', 'LAMA2', 'CNTN5', 'PDZD2', 'CUX2', 'RASGRF2', 'FAM19A1', 'LINC01378', 'CA10'],
    'L5 ET': ['COL5A2', 'FAM19A1', 'VAT1L', 'COL24A1', 'CBLN2', 'NRP1', 'PTCHD1.AS', 'NRG1', 'HOMER1', 'SLC35F3'],
    'L5 IT': ['FSTL4', 'CNTN5', 'RORB', 'FSTL5', 'IL1RAPL2', 'CHN2', 'TOX', 'CPNE4', 'CADPS2', 'POU6F2'],
    'L5/6 NP': ['TSHZ2', 'NPSR1.AS1', 'HTR2C', 'ITGA8', 'ZNF385D', 'ASIC2', 'CDH6', 'CRYM', 'NXPH2', 'CPNE4'],
    'L6 CT': ['ADAMTSL1', 'KIAA1217', 'SORCS1', 'HS3ST4', 'TRPM3', 'TOX', 'SEMA3E', 'EGFEM1P', 'MEIS2', 'SEMA5A'],
    'L6 IT': ['PTPRK', 'PDZRN4', 'CDH9', 'THEMIS', 'FSTL5', 'CDH13', 'CDH12', 'CBLN2', 'LY86.AS1', 'MLIP'],
    'L6 IT Car3': ['THEMIS', 'RNF152', 'NTNG2', 'STK32B', 'KCNMB2', 'GAS2L3', 'OLFML2B', 'POSTN', 'B3GAT2', 'NR4A2'],
    'L6b': ['HS3ST4', 'KCNMB2', 'MDFIC', 'C10orf11', 'NTM', 'CDH9', 'MARCH1', 'TLE4', 'FOXP2', 'KIAA1217'],
    'Lamp5': ['FGF13', 'PTPRT', 'PRELID2', 'GRIA4', 'RELN', 'PTCHD4', 'EYA4', 'MYO16', 'FBXL7', 'LAMP5'],
    'Micro-PVM': ['DOCK8', 'P2RY12', 'APBB1IP', 'FYB', 'PTPRC', 'TBXAS1', 'CX3CR1', 'BLNK', 'SLCO2B1', 'CSF1R'],
    'Oligo': ['PLP1', 'ST18', 'CTNNA3', 'MBP', 'MOBP', 'RNF220', 'NCKAP5', 'ENPP2', 'QKI', 'SLC44A1'],
    'OPC': ['VCAN', 'PDGFRA', 'OLIG1', 'SMOC1', 'COL9A1', 'STK32A', 'BCAS1', 'FERMT1', 'BCHE', 'ZCCHC24'],
    'Pvalb': ['ADAMTS17', 'ERBB4', 'DPP10', 'ZNF804A', 'MYO16', 'BTBD11', 'GRIA4', 'SLIT2', 'SDK1', 'PVALB'],
    'Sncg': ['CNR1', 'SLC8A1', 'ASIC2', 'CXCL14', 'MAML3', 'ADARB2', 'NPAS3', 'CNTN5', 'FSTL5', 'SNCG'],
    'Sst': ['GRIK1', 'RALYL', 'SST', 'TRHDE', 'GRID2', 'NXPH1', 'COL25A1', 'SLC8A1', 'SOX6', 'ST6GALNAC5'],
    'Sst Chodl': ['NPY', 'FAM46A', 'STAC', 'OTOF', 'NPY2R', 'CRHBP', 'ANKRD34B', 'NOS1', 'SST', 'CHODL'],
    'Vip': ['GALNTL6', 'LRP1B', 'VIP', 'GRM7', 'KCNT2', 'THSD7A', 'ERBB4', 'SYNPR', 'ADARB2', 'SLC24A3'],
    'VLMC': ['COLEC12', 'ITIH5', 'COL1A2', 'TBX18', 'EBF1', 'C7', 'COL6A2', 'SRPX2', 'FLVCR2', 'FMO2']
}

markers_mouse_cortex = {
    'Astro': ['Gpc5', 'Slc1a2', 'Slc1a3', 'Apoe', 'Wdr17', 'Plpp3', 'Rorb', 'Rmst', 'Slc4a4', 'Htra1'],
    'Endo': ['Flt1', 'Slco1a4', 'Adgrl4', 'Ly6c1', 'Slc2a1', 'Klf2', 'Mecom', 'Bsg', 'Ly6a', 'Pltp'],
    'L2/3 IT': ['Lingo2', 'Dscaml1', 'Gpc6', 'Fam19a1', 'Lrrtm4', 'Slit3', 'Cux2', 'Unc5d', 'Fstl4', 'Car10'],
    'L5 ET': ['Fam19a1', 'Tox', 'Sgcz', 'Robo1', 'Galntl6', 'Gm2164', 'Dscaml1', 'Gm28928', 'Pex5l', 'Ralyl'],
    'L5 IT': ['Car10', 'Ncam2', 'Galntl6', 'Cdh12', 'Hs3st4', 'Gpc6', 'Unc5d', 'Tenm4', 'Ptprd', 'Hs6st3'],
    'L5/6 NP': ['Tshz2', 'Gm26883', 'Nxph1', 'Vwc2l', 'Grik1', 'Olfm3', 'Kirrel3', 'Mgat4c', 'Hs6st3', 'Dcc'],
    'L6 CT': ['Hs3st4', 'Chgb', 'Garnl3', 'Kctd16', 'Zfpm2', 'Foxp2', 'Mgat4c', 'Gm28928', 'Cdh18', 'Tle4'],
    'L6 IT': ['C1ql3', 'Cdh13', 'Il1rapl2', 'Zfp804b', 'Dscaml1', 'Slit3', 'Galnt14', 'Nell2', 'Ak5', 'Cck'],
    'L6 IT Car3': ['Tfap2d', 'Oprk1', 'Smoc2', 'Car12', 'Nr4a2', 'Lxn', 'Gm32647', 'Col11a1', 'Sstr2', '6530403H02Rik'],
    'L6b': ['Ctgf', 'Cdh18', 'Hs3st4', 'Kcnab1', 'Gm28928', 'Svil', 'Inpp4b', 'Zfpm2', 'Sdk2', 'Tle4'],
    'Lamp5': ['Npy', 'Nxph1', 'Fgf13', 'Ptprm', 'Reln', 'Gad2', 'Unc5d', 'Alk', 'Unc5c', 'Lamp5'],
    'Meis2': ['Igfbpl1', 'Draxin', 'Cd1d1', 'Sp8', 'Cd24a', 'Lockd', 'St8sia2', 'Gm29260', 'Gm17750', 'Meis2'],
    'Micro-PVM': ['Inpp5d', 'Hexb', 'Tgfbr1', 'C1qa', 'Ctss', 'C1qb', 'Zfhx3', 'C1qc', 'Selplg', 'Cx3cr1'],
    'Oligo': ['Plp1', 'Mbp', 'St18', 'Prr5l', 'Mobp', 'Mal', 'Mog', 'Cldn11', 'Pde4b', 'Mag'],
    'OPC': ['Lhfpl3', 'Vcan', 'Tnr', 'Ptprz1', 'Gm4876', 'Xylt1', 'Pdgfra', 'Epn2', 'Cacng4', 'Megf11'],
    'Peri': ['Atp13a5', 'Vtn', 'Cald1', 'Ebf1', 'Abcc9', 'Dlc1', 'Pdgfrb', 'Tbx3os1', 'Pde8b', 'Slc38a11'],
    'Pvalb': ['Kcnc2', 'Erbb4', 'Sox5', 'Il1rapl2', 'Fam19a2', 'Cntnap5c', 'Lrrc4c', 'Btbd11', 'Kcnh7', 'Pvalb'],
    'SMC': ['Acta2', 'Map3k7cl', 'Gm6249', 'Myh11', 'Tagln', 'Pdlim3', 'Ephx3', 'Olfr558', 'Crip1', 'Tbx2'],
    'Sncg': ['Adarb2', 'Cnr1', 'Megf10', 'Cadps2', 'Col25a1', 'Col19a1', 'Grip1', 'Sgcd', 'Slc44a5', 'Sncg'],
    'Sst': ['Sst', 'Grin3a', 'Nxph1', 'Galntl6', 'Rgs6', 'Sox6', 'Mgat4c', 'Synpr', 'Cacna2d3', 'Grm1'],
    'Sst Chodl': ['Nos1', 'Crhbp', 'Ndst4', 'Tacr1', '9330158H04Rik', 'Chodl', 'Fndc1', 'Stac', 'Bace2', 'Gabrg1'],
    'Vip': ['Vip', 'Cacna1a', 'Adarb2', 'Limch1', 'Synpr', 'Cntn4', 'Crh', 'Sdk1', 'Igf1', 'Tac2'],
    'VLMC': ['Ptgds', 'Bnc2', 'Cped1', 'Slc7a11', 'Bmp6', 'Apod', 'Mgp', 'Eya2', 'Ranbp3l', 'Adam12']
}
