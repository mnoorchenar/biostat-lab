---
title: biostat-lab
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

<h1>🔬 BioStat Lab</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=0891b2&center=true&vCenter=true&width=700&lines=Interactive+Biostatistics+Suite;Survival+·+Mixed+Models+·+Causal+·+Bayesian;Real+Public+Datasets" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/python/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4f46e5?style=for-the-badge)](https://www.statsmodels.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

[![Live Demo](https://img.shields.io/badge/%E2%96%B6_Live_Demo-Hugging_Face_Space-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mnoorchenar/biostat-lab)
[![Source Code](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/biostat-lab)

<br/>

**🔬 BioStat Lab** — An interactive biostatistical modelling suite that runs classic methods on real, public datasets — survival analysis, linear mixed-effects models, causal inference and Bayesian regression — with every chart and estimate updating as you change the settings.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Methods & Datasets](#-methods--datasets)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#️-disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>⚕️ <b>Survival Analysis</b></td>
    <td>Kaplan-Meier curves and Cox proportional-hazards models with selectable covariates, on the Rossi recidivism data</td>
  </tr>
  <tr>
    <td>📊 <b>Linear Mixed Effects</b></td>
    <td>Random-intercept / random-slope models (MixedLM) on the sleep-deprivation study</td>
  </tr>
  <tr>
    <td>🎯 <b>Causal Inference</b></td>
    <td>1:1 nearest-neighbour propensity-score matching (UCI Diabetes) and difference-in-differences (US beer-tax fatalities)</td>
  </tr>
  <tr>
    <td>🎲 <b>Bayesian Inference</b></td>
    <td>Conjugate Bayesian linear regression with prior-to-posterior updating on UCI Diabetes</td>
  </tr>
  <tr>
    <td>🌗 <b>Light & Dark Themes</b></td>
    <td>One-click theme toggle; all Plotly charts follow the theme</td>
  </tr>
</table>

---

## 🧮 Methods & Datasets

| Tab | Methods | Dataset |
|-----|---------|---------|
| ⚕️ Survival Analysis | Kaplan-Meier, Cox PH | Rossi Recidivism (criminology) |
| 📊 Linear Mixed Effects | MixedLM | Sleep Deprivation — `sleepstudy` (neuroscience) |
| 🎯 Causal Inference | Propensity Score Matching, Difference-in-Differences | UCI Diabetes · `Fatalities` beer tax (economics) |
| 🎲 Bayesian Inference | Conjugate Bayesian Regression | UCI Diabetes (biostatistics) |

Built with **Streamlit**, **lifelines**, **statsmodels**, **scikit-learn**, **SciPy** and **Plotly**.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- Docker (optional)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/biostat-lab.git
cd biostat-lab

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py --server.port 7860
```

Open your browser at `http://localhost:7860` 🎉

---

## 🐳 Docker Deployment

```bash
docker build -t biostat-lab .
docker run -p 7860:7860 biostat-lab
```

---

## 📁 Project Structure

```
biostat-lab/
│
├── 📄 Dockerfile                 # Container definition (port 7860)
├── 📄 app.py                     # Application entry point
└── 📄 requirements.txt           # Python dependencies
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## ⚠️ Disclaimer

> [!WARNING]
> This project is developed strictly for educational purposes and does not constitute professional advice of any kind. All datasets used are either synthetically generated or publicly available. This software is provided "as is" without warranty of any kind; use at your own risk.

---

## 📜 License

Distributed under the **MIT License**.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0891b2,100:06b6d4&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![Star on GitHub](https://img.shields.io/badge/%E2%AD%90_Star-on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/biostat-lab/stargazers)
[![Fork on GitHub](https://img.shields.io/badge/%F0%9F%8D%B4_Fork-on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/biostat-lab/fork)

<sub>The name "BioStat Lab" is used purely for academic and research purposes. Any similarity to existing product names or trademarks is entirely coincidental. This project has no affiliation with any commercial entity.</sub>

</div>
