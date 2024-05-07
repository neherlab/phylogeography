

rule stable_density_one:
    output:
        "data/stable_density_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_p={p}.csv"
    shell:
        """
        python3 src/stable_density.py --N {wildcards.N} --subsampling {wildcards.p} \
                    --interaction-radius {wildcards.interaction_radius} \
                    --density-reg {wildcards.density_reg} --output {output}
        """


N_array = [1000, 500, 250]
interaction_radius_array = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
density_reg_array = [0.02, 0.05, 0.1, 0.2]
rule stable_density_all:
    input:
        expand("data/stable_density_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_p={p}.csv",
                N=N_array, interaction_radius=interaction_radius_array, density_reg=density_reg_array, p=[0.1, 0.5, 1.0])
    output:
        "data/stable_density.csv"
    run:
        import pandas as pd
        df = pd.concat([pd.read_csv(f) for f in input])
        df.to_csv(output[0], index=False)

rule stable_density_one_periodic:
    output:
        "data/stable_density_periodicBC_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_p={p}.csv"
    shell:
        """
        python3 src/stable_density.py --N {wildcards.N} --subsampling {wildcards.p} \
                    --interaction-radius {wildcards.interaction_radius} --periodic \
                    --density-reg {wildcards.density_reg} --output {output}
        """


N_array = [1000, 500, 250]
interaction_radius_array = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
density_reg_array = [0.02, 0.05, 0.1, 0.2]
rule stable_density_all_BC:
    input:
        expand("data/stable_density_periodicBC_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_p={p}.csv",
                N=N_array, interaction_radius=interaction_radius_array, density_reg=density_reg_array, p=[0.1, 0.5, 1.0])
    output:
        "data/stable_density_periodicBC.csv"
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
N_array = [1000, 500, 250]
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


rule waves_one:
    output:
        "data/waves_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_v={v}_p={p}.csv"
    shell:
        """
        python3 src/waves.py --N {wildcards.N} --subsampling {wildcards.p} \
                    --interaction-radius {wildcards.interaction_radius} \
                    --velocity {wildcards.v}\
                    --density-reg {wildcards.density_reg} --output {output}
        """

v_array = [0.0003, 0.001, 0.003, 0.01, 0.03]
N_array = [1000, 500, 250]
rule waves_all:
    input:
        expand("data/waves_subsampled_N={N}_ir={ir}_dr={dr}_v={v}_p={p}.csv",
                N=N_array, v=v_array, p=[1.0], ir=[0.05, 0.1], dr=[0.025, 0.05, 0.1, 0.2])
    output:
        "data/waves.csv"
    run:
        import pandas as pd
        df = pd.concat([pd.read_csv(f) for f in input])
        df.to_csv(output[0], index=False)


rule breathing_one:
    output:
        "data/breathing_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_T={T}_p={p}.csv"
    shell:
        """
        python3 src/breathing.py --N {wildcards.N} --subsampling {wildcards.p} \
                    --interaction-radius {wildcards.interaction_radius} \
                    --period {wildcards.T}\
                    --density-reg {wildcards.density_reg} --output {output}
        """

T_array = [10,50,100,200, 500]
N_array = [500, 250]
rule breathing_all:
    input:
        expand("data/breathing_subsampled_N={N}_ir={ir}_dr={dr}_T={T}_p={p}.csv",
                N=N_array, T=T_array, p=[1.0], ir=[0.05, 0.1], dr=[0.025, 0.05, 0.1, 0.2])
    output:
        "data/breathing.csv"
    run:
        import pandas as pd
        df = pd.concat([pd.read_csv(f) for f in input])
        df.to_csv(output[0], index=False)

rule seasaw_one:
    output:
        "data/seasaw_subsampled_N={N}_ir={interaction_radius}_dr={density_reg}_T={T}_p={p}.csv"
    shell:
        """
        python3 src/seasaw.py --N {wildcards.N} --subsampling {wildcards.p} \
                    --interaction-radius {wildcards.interaction_radius} \
                    --period {wildcards.T}\
                    --density-reg {wildcards.density_reg} --output {output}
        """

T_array = [10,50,100,200, 500]
N_array = [1000, 500, 250]
rule seasaw_all:
    input:
        expand("data/seasaw_subsampled_N={N}_ir={ir}_dr={dr}_T={T}_p={p}.csv",
                N=N_array, T=T_array, p=[0.1, 1.0], ir=[0.05, 0.1], dr=[0.025, 0.05, 0.1, 0.2])
    output:
        "data/seasaw.csv"
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
