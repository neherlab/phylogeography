

rule inflated_diffusion_one:
    output:
        "data/inflated_diffusion_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_p={p}.csv"
    shell:
        """
        python3 src/inflated_diffusion.py --N {wildcards.N} --subsampling {wildcards.p} \
                    --interaction-radius {wildcards.interaction_radius} \
                    --density-reg {wildcards.density_reg} --output {output}
        """


N_array = [500]
interaction_radius_array = [0.02, 0.05, 0.1, 0.2, 0.5]
density_reg_array = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
rule inflated_diffusion_all:
    input:
        expand("data/inflated_diffusion_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_p={p}.csv",
                N=N_array, interaction_radius=interaction_radius_array, density_reg=density_reg_array, p=0.1)
    output:
        "data/inflated_diffusion_p=0.1.csv"
    run:
        import pandas as pd
        df = pd.concat([pd.read_csv(f) for f in input])
        df.to_csv(output[0], index=False)

