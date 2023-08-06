

rule inflated_diffusion_one:
    output:
        "data/inflated_diffusion_N={N}_ir={interaction_radius}_dr={density_reg}.csv"
    shell:
        """
        python3 src/inflated_diffusion.py --N {N} --interaction-radius {interaction_radius} --density-reg {density_reg} --output {output}
        """


N = [500]
interaction_radius = [0.02, 0.05, 0.1, 0.2, 0.5]
density_reg = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
rule inflated_diffusion_all:
    input:
        expand("data/inflated_diffusion_N={N}_ir={interaction_radius}_dr={density_reg}.csv", N=N, interaction_radius=interaction_radius, density_reg=density_reg)
    output:
        "data/inflated_diffusion.csv"
    run:
        import pandas as pd
        df = pd.concat([pd.read_csv(f) for f in input])
        df.to_csv(output[0], index=False)

