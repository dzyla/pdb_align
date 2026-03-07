import re

with open("pdb_align/core.py", "r") as f:
    text = f.read()

# sequence_independent_alignment_joined_v2 parameter update
new_func = """def sequence_independent_alignment_joined_v2(
    file_ref: str, file_mob: str,
    chains_ref: Optional[List[Union[str,int]]]=None,
    chains_mob: Optional[List[Union[str,int]]]=None,
    method:str="auto",
    shape_nbins:int=24, shape_gap_penalty:float=2.0, shape_band_frac:float=0.20,
    inlier_rmsd_cut:float=3.0, inlier_quantile:float=0.85, refinement_iters:int=2,
    atoms: str = "CA", min_b_factor: float = 0.0
)->AlignmentResultSF:"""

text = re.sub(
    r'def sequence_independent_alignment_joined_v2\(\n\s+file_ref: str, file_mob: str,\n\s+chains_ref: Optional\[List\[Union\[str,int\]\]\]=None,\n\s+chains_mob: Optional\[List\[Union\[str,int\]\]\]=None,\n\s+method:str="auto",\n\s+shape_nbins:int=24, shape_gap_penalty:float=2\.0, shape_band_frac:float=0\.20,\n\s+inlier_rmsd_cut:float=3\.0, inlier_quantile:float=0\.85, refinement_iters:int=2,\n\s+atoms: str = "CA"\n\)->AlignmentResultSF:',
    new_func,
    text
)

text = text.replace("ref_infos=_extract_ca_infos(ref_struct, ref_ids)", "ref_infos=_extract_ca_infos(ref_struct, ref_ids, min_b_factor)")
text = text.replace("mob_infos=_extract_ca_infos(mob_struct, mob_ids)", "mob_infos=_extract_ca_infos(mob_struct, mob_ids, min_b_factor)")


with open("pdb_align/core.py", "w") as f:
    f.write(text)
