

rule inflated_diffusion_one:
    output:
        "data/inflated_diffusion_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_p={p}.csv"
    shell:
        """
        python3 src/inflated_diffusion.py --N {wildcards.N} --subsampling {wildcards.p} \
                    --interaction-radius {wildcards.interaction_radius} \
                    --density-reg {wildcards.density_reg} --output {output}
        """


N_array = [500, 1000]
interaction_radius_array = [0.05, 0.1, 0.2, 0.3, 0.5]
density_reg_array = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
rule inflated_diffusion_all:
    input:
        expand("data/inflated_diffusion_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_p={p}.csv",
                N=N_array, interaction_radius=interaction_radius_array, density_reg=density_reg_array, p=[0.1, 0.5, 1.0])
    output:
        "data/inflated_diffusion.csv"
    run:
        import pandas as pd
        df = pd.concat([pd.read_csv(f) for f in input])
        df.to_csv(output[0], index=False)

rule habitat_diffusion_one:
    output:
        "data/habitat_diffusion_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_T={T}_p={p}.csv"
    shell:
        """
        python3 src/habitat_shifts.py --N {wildcards.N} --subsampling {wildcards.p} \
                    --interaction-radius {wildcards.interaction_radius} \
                    --period {wildcards.T}\
                    --density-reg {wildcards.density_reg} --output {output}
        """

T_array = [10,50,100,200, 500]
N_array = [500,1000]
rule habitat_diffusion_all:
    input:
        expand("data/habitat_diffusion_subsampled_N={N}_ir={ir}_dr={dr}_T={T}_p={p}.csv",
                N=N_array, T=T_array, p=[0.1, 0.5, 1.0], ir=[0.05, 0.1], dr=[0.1, 0.2])
    output:
        "data/habitat_diffusion.csv"
    run:
        import pandas as pd
        df = pd.concat([pd.read_csv(f) for f in input])
        df.to_csv(output[0], index=False)



rule plot_inflated_diffusion:
    input:
        "data/inflated_diffusion.csv"
    output:
        heterogeneity = "manuscript/figures/density_reg_heterogeneity.pdf",
        dest = "manuscript/figures/density_reg_dest.pdf",
        tmrca = "manuscript/figures/density_reg_tmrca.pdf"
    shell:
        """
        python3 src/plot_inflated_diffusion.py --data {input} \
            --output-heterogeneity {output.heterogeneity} \
            --output-diffusion {output.dest} \
            --output-tmrca {output.tmrca}
        """

rule plot_habitat_diffusion:
    input:
        "data/habitat_diffusion.csv"
    output:
        illustration = "manuscript/figures/habitats.png",
        dest = "manuscript/figures/habitat_diffusion.pdf",
        tmrca = "manuscript/figures/habitat_tmrca.pdf"
    shell:
        """
        python3 src/plot_habitat_diffusion.py --data {input} \
            --output-diffusion {output.dest} \
            --output-tmrca {output.tmrca} \
            --illustration {output.illustration}
        """
