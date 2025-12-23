🛡️ Detecting Face Morphing Attacks for Secure Digital Transactions

This repository contains the implementation of my Final Year Project (FYP) titled:

“Detecting Face Morphing Attacks for Secure Digital Transactions”

The project focuses on improving the security of eKYC and digital identity verification systems by detecting face morphing attacks, where two identities are blended into a single facial image to bypass biometric authentication.

📌 Project Overview

Face morphing attacks pose a serious threat to biometric systems used in digital banking, online onboarding, and identity verification. Traditional face recognition systems often fail to detect whether an input image has been manipulated.

This project investigates both:

Single-image Morphing Attack Detection (S-MAD)

Differential Morphing Attack Detection (D-MAD)

and proposes a hybrid framework that combines the strengths of both approaches.

🎯 Objectives

To study the vulnerability of face recognition systems to morphing attacks

To implement S-MAD using deep CNN-based feature learning

To implement D-MAD using a shared-encoder Siamese architecture with cosine similarity

To design a hybrid decision framework for robust morph detection

To ensure leakage-free and fair evaluation using identity-aware data splitting

🧠 Methodology
1. Single-image MAD (S-MAD)

Uses a single facial image as input

Learns texture inconsistencies and morphing artifacts

Implemented using a deep CNN backbone (e.g. EfficientNet)

2. Differential MAD (D-MAD)

Takes a pair of images (ID vs selfie) as input

Uses a shared encoder to extract embeddings

Computes cosine similarity to detect morphing inconsistencies

3. Hybrid Framework

Combines S-MAD and D-MAD decisions

Improves robustness against both high-quality morphs and identity mismatches

📂 Datasets

This project uses publicly available face morphing datasets:

FEI Face Database

FRLL Face Morphing Dataset

⚠️ Data Splitting Strategy (Important)

To prevent data leakage:

Identity-aware splitting is applied

No image file or identity appears in more than one split

Training, validation, and testing sets are strictly separated

This ensures realistic and trustworthy evaluation results.

👩‍🎓 Author

Name: 郭乙慈
Program: Bachelor of Computer Science (Data Science)
University: University of Malaya
Supervisor: Dr. Hoo Wai Lam
