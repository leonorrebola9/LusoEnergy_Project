from importlib.metadata import version, PackageNotFoundError

pacotes = ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "requests",
           "jupyter", "river", "plotly", "streamlit", "holidays"]
linhas = []
for p in pacotes:
    try:
        linhas.append(f"{p}=={version(p)}")
    except PackageNotFoundError:
        print("não instalado (ainda):", p)

with open("requirements.txt", "w") as f:
    f.write("\n".join(linhas) + "\n")
print(open("requirements.txt").read())