import streamlit as st

st.set_page_config(
    page_title="LLM Benchmarks"
)

st.header("LLM Benchmarks")

benchmarks = ["AI2 ARC", "AI2d", "Big Bench Hard", "BLINK", "ChatDoctor-iClinq", "Competitive Math", "GPQA", "GSM8K", "GSM Plus", "MATH", "MathV360K", "MathVision", "MathVista", "Medical Meadow Flashcards", "Medical Meadow MedQA", "Medical Meadow Wikidoc Patient", "Medical Questions", "MedQA", "MedQA USMLE-4", "Medtrinity25M", "MetaMathQA", "MMLU", "MMLU Pro", "MMMU", "Olympic Arena", "RealWorldQA", "Scibench", "ScienceQA", "SLAKE", "Symptom Disease", "TruthfulQA", "VisIT Bench", "Winogrande"]

benchmark_articles = {
    "AI2 ARC": """
    ## AI2 Reasoning Challenge (ARC)

    The AI2 Reasoning Challenge (ARC) is a dataset for evaluating machine learning models' reasoning abilities in elementary science. Developed by the Allen Institute for Artificial Intelligence, it consists of multiple-choice questions from US standardized tests for grades 3-9.

    ### Features
    - 7,787 grade-school level, multiple-choice science questions
    - Covers various science topics and reasoning skills
    - Questions require more than simple information retrieval

    ### Relevance
    ARC assesses AI models' scientific reasoning and ability to apply knowledge to new situations, pushing towards advanced scientific problem-solving in AI.

    ### Citation
    Clark, P., et al. (2018). Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457.
    """,

    "AI2D": """
    # AI2 Diagram (AI2D)

    AI2D is a dataset and benchmark for understanding scientific diagrams. Created by the Allen Institute for Artificial Intelligence, it evaluates AI models' ability to interpret and reason about visual information in scientific contexts.

    ### Features
    - Science diagrams with associated questions and answers
    - Various diagram types: parts of objects, cycles, flow charts, etc.
    - Requires multi-modal reasoning combining visual and textual information

    ### Relevance
    AI2D is crucial for developing AI systems that can understand visual representations in educational and scientific contexts, advancing multi-modal AI capabilities.

    ### Citation
    Kembhavi, A., et al. (2016). A Diagram Is Worth A Dozen Images. In Computer Vision – ECCV 2016.
    """,

    "Big Bench Hard": """
    # Big Bench Hard

    Big Bench Hard is a subset of the larger BIG-bench project, focusing on particularly challenging tasks for AI systems. It aims to push the limits of language models and identify areas needing significant improvement.

    ### Features
    - Curated set of difficult tasks from the BIG-bench collection
    - Covers a wide range of cognitive abilities and knowledge domains
    - Includes tasks requiring complex reasoning, creativity, and specialized knowledge

    ### Relevance
    Big Bench Hard serves as a stress test for state-of-the-art AI models, highlighting areas where human-level performance is still out of reach and tracking progress in tackling hard problems.

    ### Citation
    Srivastava, A., et al. (2022). Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv:2206.04615.
    """,

    "BLINK": """
    # Benchmarking LInk prediction on Knowledge graphs (BLINK)

    BLINK is a benchmark dataset for evaluating entity linking systems, particularly in the context of knowledge graphs. It tests the ability to connect entity mentions in text to their corresponding knowledge base entries.

    ## Features
    - Large-scale dataset with diverse entity mentions
    - Covers multiple domains and languages
    - Includes both popular and rare entities to test robustness

    ## Relevance
    BLINK is crucial for assessing AI systems' ability to understand and connect textual mentions to structured knowledge, essential for information retrieval and knowledge graph completion tasks.

    ## Citation
    Wu, L., et al. (2020). Scalable Zero-shot Entity Linking with Dense Entity Retrieval. In Proceedings of EMNLP 2020.
    """,

    "ChatDoctor-iClinq": """
    # ChatDoctor-iClinq

    ChatDoctor-iClinq is a benchmark for evaluating AI models' performance in medical dialogue and clinical decision-making scenarios. It simulates patient-doctor interactions to assess AI's ability in gathering information, providing medical advice, and making diagnoses.

    ## Features
    - Collection of realistic medical conversation scenarios
    - Covers a wide range of medical conditions and patient profiles
    - Assesses both medical knowledge and communication skills

    ## Relevance
    This benchmark is important for developing AI systems that can assist in healthcare settings, supporting medical professionals and improving patient care through accurate and empathetic communication.

    ## Citation
    Refer to official ChatDoctor-iClinq documentation or recent publications in medical AI for the most up-to-date information.
    """,

    "Competitive Math": """
    # Competitive Math

    The Competitive Math benchmark evaluates AI models' ability to solve complex mathematical problems typically found in high-level mathematics competitions. These problems require advanced problem-solving skills, creative thinking, and deep mathematical understanding.

    ## Features
    - Collection of challenging math problems from various competitions
    - Covers algebra, geometry, number theory, combinatorics, and more
    - Requires both computational skills and mathematical insight

    ## Relevance
    This benchmark assesses AI systems' capabilities in advanced mathematical reasoning, pushing boundaries in fields requiring high-level problem-solving and potentially leading to breakthroughs in automated theorem proving and mathematical discovery.

    ## Citation
    Refer to papers discussing AI performance on mathematical olympiad problems or competitive programming challenges for related work in this area.
    """,

    "GPQA": """
    # General Physics Question Answering (GPQA)

    GPQA is a benchmark designed to evaluate AI models' ability to answer questions in the domain of general physics. It tests the understanding of fundamental physics concepts and problem-solving skills.

    ## Features
    - Diverse set of physics questions covering mechanics, thermodynamics, electromagnetism, etc.
    - Requires both conceptual understanding and mathematical problem-solving
    - Includes questions of varying difficulty levels

    ## Relevance
    GPQA is crucial for assessing AI systems' capabilities in scientific reasoning and application of physics principles, which is essential for advancing AI in scientific and engineering domains.

    ## Citation
    Refer to the official GPQA documentation or recent publications in AI for physics for the most up-to-date information.
    """,

    "GSM8K": """
    # Grade School Math 8K (GSM8K)

    GSM8K is a dataset of 8,500 high-quality linguistically diverse grade school math word problems. It is designed to test language models' ability to solve multi-step mathematical reasoning problems.

    ## Features
    - 8,500 grade school math problems
    - Requires multi-step reasoning
    - Diverse problem formulations and linguistic structures

    ## Relevance
    GSM8K is important for evaluating AI models' mathematical reasoning capabilities in the context of natural language understanding, bridging the gap between language processing and mathematical problem-solving.

    ## Citation
    Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.
    """,

    "GSM Plus": """
    # Grade School Math Plus (GSM Plus)

    GSM Plus is an extended version of the GSM8K dataset, offering a broader range of grade school math problems. It aims to provide a more comprehensive evaluation of AI models' mathematical reasoning abilities.

    ## Features
    - Expanded set of grade school math problems
    - Includes more diverse problem types and difficulty levels
    - Focuses on multi-step reasoning and problem-solving strategies

    ## Relevance
    GSM Plus helps in assessing the scalability and generalization capabilities of AI models in mathematical reasoning, providing insights into their potential for educational applications and general problem-solving.

    ## Citation
    Refer to the official GSM Plus documentation or recent publications for the most up-to-date information.
    """,

    "MATH": """
    # MATH Dataset

    The MATH dataset is a collection of mathematics problems designed to evaluate AI models' advanced mathematical reasoning capabilities. It covers a wide range of mathematical topics and difficulty levels.

    ## Features
    - Diverse set of mathematics problems from various fields
    - Includes problems from high school to undergraduate level
    - Requires both procedural knowledge and creative problem-solving

    ## Relevance
    MATH is crucial for pushing the boundaries of AI in advanced mathematical reasoning, with potential applications in automated theorem proving, mathematical research, and STEM education.

    ## Citation
    Hendrycks, D., et al. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. arXiv:2103.03874.
    """,

    "MathV360K": """
    # MathV360K

    MathV360K is a large-scale dataset focusing on visual mathematics problems. It combines mathematical reasoning with visual understanding, presenting problems that require interpretation of diagrams, graphs, and other visual elements.

    ## Features
    - 360,000 visual mathematics problems
    - Covers various math topics with visual components
    - Requires integration of visual and mathematical reasoning

    ## Relevance
    MathV360K is important for developing AI systems that can handle real-world mathematical problems involving visual elements, bridging the gap between computer vision and mathematical reasoning.

    ## Citation
    Refer to the official MathV360K documentation or recent publications in visual mathematics for AI for the most up-to-date information.
    """,

    "MathVision": """
    # MathVision

    MathVision is a benchmark designed to test AI models' ability to solve mathematical problems presented in visual formats. It focuses on the intersection of computer vision and mathematical reasoning.

    ## Features
    - Collection of mathematics problems with visual components
    - Includes diagrams, graphs, geometric figures, and other visual representations
    - Requires both visual understanding and mathematical problem-solving skills

    ## Relevance
    MathVision is crucial for developing AI systems capable of interpreting and solving real-world mathematical problems that often involve visual elements, advancing the field of AI in education and scientific visualization.

    ## Citation
    Refer to the official MathVision documentation or recent publications in visual mathematics and AI for the most up-to-date information.
    """,

    "MathVista": """
    # MathVista

    MathVista is a comprehensive benchmark for evaluating AI models' capabilities in visual mathematical reasoning. It combines advanced mathematics with complex visual understanding tasks.

    ## Features
    - Diverse set of visual mathematics problems
    - Covers a wide range of mathematical topics and visual representations
    - Requires sophisticated integration of visual processing and mathematical reasoning

    ## Relevance
    MathVista is essential for pushing the boundaries of AI in handling complex, real-world mathematical scenarios that involve visual interpretation, with applications in fields like scientific research, engineering, and data visualization.

    ## Citation
    Refer to the official MathVista documentation or recent publications in visual mathematics and AI for the most up-to-date information.
    """,

    "Medical Meadow Flashcards": """
    # Medical Meadow Flashcards

    Medical Meadow Flashcards is a benchmark designed to evaluate AI models' ability to learn and recall medical information in a flashcard-style format. It simulates the learning process of medical students.

    ## Features
    - Comprehensive set of medical flashcards covering various topics
    - Tests both factual recall and application of medical knowledge
    - Mimics the spaced repetition learning technique

    ## Relevance
    This benchmark is crucial for developing AI systems that can assist in medical education and training, potentially revolutionizing how medical knowledge is acquired and retained.

    ## Citation
    Refer to the official Medical Meadow documentation or recent publications in medical education AI for the most up-to-date information.
    """,

    "Medical Meadow MedQA": """
    # Medical Meadow MedQA

    Medical Meadow MedQA is a benchmark focused on evaluating AI models' ability to answer complex medical questions. It simulates the types of questions encountered in medical practice and exams.

    ## Features
    - Diverse set of medical questions covering various specialties
    - Includes case-based scenarios and diagnostic challenges
    - Requires integration of medical knowledge and clinical reasoning

    ## Relevance
    This benchmark is important for assessing AI systems' potential in supporting medical decision-making and education, pushing towards more advanced AI assistants in healthcare.

    ## Citation
    Refer to the official Medical Meadow documentation or recent publications in medical AI for the most up-to-date information.
    """,

    "Medical Meadow Wikidoc Patient": """
    # Medical Meadow Wikidoc Patient

    Medical Meadow Wikidoc Patient is a benchmark that evaluates AI models' ability to understand and generate patient-friendly medical information. It focuses on translating complex medical concepts into accessible language.

    ## Features
    - Collection of medical topics to be explained in layman's terms
    - Covers a wide range of medical conditions, treatments, and procedures
    - Assesses clarity, accuracy, and appropriateness of explanations for patients

    ## Relevance
    This benchmark is crucial for developing AI systems that can assist in patient education and improve health literacy, potentially enhancing patient-doctor communication and overall healthcare outcomes.

    ## Citation
    Refer to the official Medical Meadow documentation or recent publications in medical communication AI for the most up-to-date information.
    """,

    "Medical Questions": """
    # Medical Questions

    The Medical Questions benchmark is designed to evaluate AI models' ability to answer a wide range of medical queries. It covers various aspects of healthcare, from basic anatomy to complex diagnostic scenarios.

    ## Features
    - Diverse set of medical questions across different specialties
    - Includes both factual and scenario-based questions
    - Tests depth and breadth of medical knowledge

    ## Relevance
    This benchmark is essential for assessing AI systems' potential in supporting medical professionals, medical education, and general health information dissemination.

    ## Citation
    Refer to recent publications in medical AI and question-answering systems for the most up-to-date information on medical question benchmarks.
    """,

    "MedQA": """
    # MedQA

    MedQA is a comprehensive benchmark for evaluating AI models' performance on medical question answering tasks. It aims to assess the ability of AI systems to understand and respond to complex medical queries.

    ## Features
    - Large-scale dataset of medical questions and answers
    - Covers various medical specialties and question types
    - Includes both multiple-choice and open-ended questions

    ## Relevance
    MedQA is crucial for developing AI systems that can assist healthcare professionals in decision-making, medical research, and patient care, potentially improving the efficiency and accuracy of medical information retrieval.

    ## Citation
    Jin, D., et al. (2021). What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams. Applied Sciences, 11(14), 6421.
    """,

    "MedQA USMLE-4": """
    # MedQA USMLE-4

    MedQA USMLE-4 is a specialized benchmark based on the United States Medical Licensing Examination Step 4. It evaluates AI models' ability to handle complex, clinical scenario-based questions typical of advanced medical licensing exams.

    ## Features
    - High-difficulty medical questions based on USMLE Step 4 format
    - Focuses on clinical decision-making and patient management
    - Requires integration of medical knowledge across multiple domains

    ## Relevance
    This benchmark is crucial for assessing AI systems' potential in advanced medical reasoning and decision-making, pushing the boundaries of AI in healthcare towards expert-level performance.

    ## Citation
    Refer to official USMLE documentation and recent publications in medical AI for the most up-to-date information on this benchmark.
    """,

    "Medtrinity25M": """
    # Medtrinity25M

    Medtrinity25M is a large-scale medical knowledge benchmark consisting of 25 million data points. It aims to evaluate AI models' comprehensive understanding of medical concepts, relationships, and reasoning.

    ## Features
    - 25 million medical data points covering diverse medical topics
    - Includes various types of medical knowledge and relationships
    - Tests both breadth and depth of medical understanding

    ## Relevance
    Medtrinity25M is essential for developing and evaluating AI systems with broad and deep medical knowledge, potentially leading to more comprehensive and reliable AI assistants in healthcare.

    ## Citation
    Refer to the official Medtrinity25M documentation or recent publications in large-scale medical AI benchmarks for the most up-to-date information.
    """,

    "MetaMathQA": """
    # MetaMathQA

    MetaMathQA is a benchmark designed to evaluate AI models' ability to solve meta-level mathematical problems. It focuses on questions that require not just mathematical computation, but also understanding of mathematical concepts and methodologies.

    ## Features
    - Collection of meta-level mathematics questions
    - Covers topics in mathematical logic, proof strategies, and problem-solving approaches
    - Requires higher-order thinking and mathematical reasoning skills

    ## Relevance
    MetaMathQA is crucial for developing AI systems capable of advanced mathematical thinking, potentially leading to breakthroughs in automated mathematical research and education.

    ## Citation
    Refer to recent publications in AI for mathematics and meta-mathematical reasoning for the most up-to-date information on this benchmark.
    """,

    "MMLU": """
    # Massive Multitask Language Understanding (MMLU)

    MMLU is a comprehensive benchmark designed to test AI models' general knowledge and problem-solving abilities across a wide range of subjects. It covers topics from elementary to professional-level expertise.

    ## Features
    - 57 tasks spanning STEM, humanities, social sciences, and more
    - Multiple-choice questions of varying difficulty
    - Requires broad knowledge and reasoning capabilities

    ## Relevance
    MMLU is crucial for evaluating the general intelligence and knowledge breadth of AI models, providing insights into their potential for real-world applications across various domains.

    ## Citation
    Hendrycks, D., et al. (2020). Measuring Massive Multitask Language Understanding. arXiv:2009.03300.
    """,

    "MMLU Pro": """
    # MMLU Pro

    MMLU Pro is an advanced version of the MMLU benchmark, focusing on professional and specialized knowledge domains. It aims to evaluate AI models' capabilities in handling expert-level questions across various fields.

    ## Features
    - Advanced questions in professional fields like law, medicine, and engineering
    - Requires deep domain-specific knowledge and reasoning
    - Tests the limits of AI models in specialized areas

    ## Relevance
    MMLU Pro is essential for assessing AI systems' potential in professional and specialized contexts, pushing the boundaries of AI applications in expert domains.

    ## Citation
    Refer to the official MMLU Pro documentation or recent publications in advanced AI benchmarking for the most up-to-date information.
    """,

    "MMMU": """
    # Massive Multi-discipline Multimodal Understanding (MMMU)

    MMMU is a benchmark designed to evaluate AI models' ability to understand and reason across multiple disciplines using various modalities (text, images, diagrams, etc.).

    ## Features
    - Tasks spanning multiple academic and professional disciplines
    - Incorporates multiple modalities including text, images, and diagrams
    - Requires integration of knowledge across different formats and fields

    ## Relevance
    MMMU is crucial for developing AI systems capable of handling complex, real-world scenarios that involve multiple disciplines and types of information.

    ## Citation
    Refer to recent publications in multimodal AI and interdisciplinary benchmarks for the most up-to-date information on MMMU.
    """,

    "Olympic Arena": """
    # Olympic Arena

    Olympic Arena is a benchmark designed to test AI models' problem-solving abilities in the context of olympiad-level challenges across various disciplines, particularly in science and mathematics.

    ## Features
    - Collection of high-difficulty problems from academic olympiads
    - Covers fields like mathematics, physics, chemistry, and informatics
    - Requires advanced problem-solving skills and creative thinking

    ## Relevance
    This benchmark is crucial for pushing the limits of AI in tackling extremely challenging problems, simulating the highest levels of human academic competition.

    ## Citation
    Refer to recent publications on AI for olympiad-level problem-solving for the most up-to-date information on this benchmark.
    """,

    "RealWorldQA": """
    # RealWorldQA

    RealWorldQA is a benchmark designed to evaluate AI models' ability to answer questions about real-world scenarios and practical situations. It focuses on common sense reasoning and practical knowledge application.

    ## Features
    - Questions based on everyday scenarios and real-world problems
    - Covers a wide range of topics relevant to daily life
    - Requires both factual knowledge and practical reasoning

    ## Relevance
    RealWorldQA is important for assessing AI systems' potential in handling practical, real-life situations, which is crucial for developing AI assistants that can provide useful advice in everyday contexts.

    ## Citation
    Refer to recent publications in AI for common sense reasoning and practical question answering for the most up-to-date information on this benchmark.
    """,

    "Scibench": """
    # Scibench

    Scibench is a comprehensive benchmark for evaluating AI models' understanding and reasoning capabilities in scientific domains. It covers a broad range of scientific disciplines and concepts.

    ## Features
    - Questions spanning various scientific fields (physics, chemistry, biology, etc.)
    - Includes both theoretical concepts and practical applications
    - Requires deep scientific knowledge and reasoning skills

    ## Relevance
    Scibench is crucial for assessing AI systems' potential in scientific research, education, and problem-solving, pushing towards more advanced AI assistants in STEM fields.

    ## Citation
    Refer to recent publications in AI for scientific reasoning and benchmarking for the most up-to-date information on Scibench.
    """,

    "ScienceQA": """
    # ScienceQA

    ScienceQA is a benchmark focused on evaluating AI models' ability to answer scientific questions across various disciplines. It aims to test both factual recall and scientific reasoning.

    ## Features
    - Diverse set of science questions from elementary to advanced levels
    - Covers multiple scientific disciplines
    - Includes both factual and conceptual questions

    ## Relevance
    ScienceQA is important for developing AI systems that can assist in science education and research, potentially enhancing scientific literacy and problem-solving capabilities.

    ## Citation
    Refer to recent publications in AI for science education and question answering for the most up-to-date information on ScienceQA.
    """,

    "SLAKE": """
    # SLAKE (Structured Language Knowledge Evaluation)

    SLAKE is a benchmark designed to evaluate AI models' ability to understand and reason about structured language and knowledge representations. It focuses on tasks involving logical reasoning and knowledge manipulation.

    ## Features
    - Tasks involving structured language and knowledge graphs
    - Requires logical reasoning and inference capabilities
    - Tests understanding of relationships and hierarchies in knowledge

    ## Relevance
    SLAKE is crucial for developing AI systems capable of handling complex, structured information and performing logical reasoning tasks, which is essential for advanced AI applications in fields like database management and expert systems.

    ## Citation
    Refer to recent publications in AI for structured knowledge representation and reasoning for the most up-to-date information on SLAKE.
    """,

    "Symptom Disease": """
    # Symptom Disease

    The Symptom Disease benchmark is designed to evaluate AI models' ability to associate symptoms with potential diseases. It tests the capability of AI systems in basic medical diagnosis and health information processing.

    ## Features
    - Dataset of symptoms and corresponding diseases
    - Includes both common and rare medical conditions
    - Tests pattern recognition and medical knowledge application

    ## Relevance
    This benchmark is important for developing AI systems that can assist in preliminary medical diagnosis and health information dissemination, potentially improving early disease detection and health awareness.

    ## Citation
    Refer to recent publications in AI for medical diagnosis and symptom-disease association for the most up-to-date information on this benchmark.
    """,

    "TruthfulQA": """
    # TruthfulQA

    TruthfulQA is a benchmark designed to evaluate the truthfulness and factual accuracy of AI language models. It focuses on the model's ability to provide honest and correct information, even in scenarios where humans might be biased or misinformed.

    ## Features
    - Questions designed to probe for common misconceptions and false beliefs
    - Covers a wide range of topics including science, history, and current events
    - Assesses both factual accuracy and the ability to avoid generating false information

    ## Relevance
    TruthfulQA is crucial for developing AI systems that can provide reliable and truthful information, which is essential for building trust in AI-assisted information services and decision-making tools.

    ## Citation
    Lin, S., et al. (2021). TruthfulQA: Measuring How Models Mimic Human Falsehoods. arXiv:2109.07958.
    """,

    "VisIT Bench": """
    # Visual Instruction Tuning Benchmark (VisIT Bench)

    VisIT Bench is a benchmark designed to evaluate AI models' ability to understand and follow visual instructions. It focuses on the integration of visual perception and language understanding in task completion.

    ## Features
    - Tasks involving visual instructions and image-based problem-solving
    - Covers a range of scenarios from simple object manipulation to complex scene interpretation
    - Requires coordination between visual processing and language comprehension

    ## Relevance
    VisIT Bench is important for developing AI systems capable of understanding and executing tasks based on visual instructions, which is crucial for applications in robotics, augmented reality, and visual-based AI assistants.

    ## Citation
    Refer to recent publications in visual instruction tuning and multimodal AI benchmarks for the most up-to-date information on VisIT Bench.
    """,

    "Winogrande": """
    # Winogrande

    Winogrande is an advanced benchmark for commonsense reasoning in AI, based on the Winograd Schema Challenge. It presents a large-scale dataset of sentence completion tasks that require human-like reasoning capabilities.

    ## Features
    - 44k sentence completion problems
    - Requires understanding of context and implicit knowledge
    - Designed to be challenging for statistical learning methods

    ## Relevance
    Winogrande is crucial for evaluating and improving AI models' commonsense reasoning abilities, which is essential for developing more human-like AI systems capable of understanding nuanced contexts in natural language.

    ## Citation
    Sakaguchi, K., et al. (2021). WINOGRANDE: An Adversarial Winograd Schema Challenge at Scale. Communications of the ACM, 64(9), 99-106.
    """
}

for benchmark, article in benchmark_articles.items():
    st.markdown(article)
    st.divider()

