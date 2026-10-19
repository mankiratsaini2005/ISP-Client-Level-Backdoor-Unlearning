import torch
init_sd = torch.load("results_phase2/iid_attack_logged/initial_model.pt", map_location="cpu", weights_only=True)
keys = [k for k in init_sd.keys() if "num_batches_tracked" in k]

for config in ["iid_Ecal1_v2", "iid_Ecal2_v2", "noniid_Ecal1_v2", "noniid_Ecal2_v2"]:
    print(f"[{config}]")
    final_sd = torch.load(f"results_phase2/federaser/{config}/final_model.pt", map_location="cpu", weights_only=True)
    all_unchanged = True
    for k in keys:
        v_init = init_sd[k].item()
        v_final = final_sd[k].item()
        print(f"  {k}: Initial = {v_init}, Final = {v_final}")
        if v_init != v_final: all_unchanged = False
    print(f"  All unchanged? {all_unchanged}")

