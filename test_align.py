from struct_pair_align import sequence_independent_alignment_joined_v2, _parse_path, _extract_ca_infos

print("Testing sequence_independent_alignment_joined_v2 (auto)")
try:
    res_auto = sequence_independent_alignment_joined_v2("1CRN.pdb", "2CRN.pdb", chains_ref=["A"], chains_mob=["A"], method="auto")
    print(f"Auto RMSD: {res_auto.rmsd}, Method chosen: {res_auto.method}, Keep Pairs: {res_auto.kept_pairs}")
except Exception as e:
    print(f"Error (auto): {e}")

print("Testing sequence_independent_alignment_joined_v2 (shape)")
try:
    res_shape = sequence_independent_alignment_joined_v2("1CRN.pdb", "2CRN.pdb", chains_ref=["A"], chains_mob=["A"], method="shape")
    print(f"Shape RMSD: {res_shape.rmsd}, Method chosen: {res_shape.method}, Matrix shape: {res_shape.shift_matrix.shape if res_shape.shift_matrix is not None else None}")
except Exception as e:
    print(f"Error (shape): {e}")

print("Testing sequence_independent_alignment_joined_v2 (window)")
try:
    res_window = sequence_independent_alignment_joined_v2("1CRN.pdb", "2CRN.pdb", chains_ref=["A"], chains_mob=["A"], method="window")
    print(f"Window RMSD: {res_window.rmsd}, Method chosen: {res_window.method}, Scores len: {len(res_window.shift_scores) if res_window.shift_scores is not None else None}")
except Exception as e:
    print(f"Error (window): {e}")
