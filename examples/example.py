from llama_cpp import Llama
import json
import sys

questions = [
    """Shortly after undergoing a bipolar prosthesis for a displaced femoral neck fracture of the left hip acquired after a fall the day before, an 80-year-old woman suddenly develops dyspnea. The surgery under general anesthesia with sevoflurane was uneventful, lasting 98 minutes, during which the patient maintained oxygen saturation readings of 100% on 8 L of oxygen. She has a history of hypertension, osteoporosis, and osteoarthritis of her right knee. Her medications include ramipril, naproxen, ranitidine, and a multivitamin. She appears cyanotic, drowsy, and is oriented only to person. Her temperature is 38.6\u00b0C (101.5\u00b0F), pulse is 135/minute, respirations are 36/min, and blood pressure is 155/95 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 81%. There are several scattered petechiae on the anterior chest wall. Laboratory studies show a hemoglobin concentration of 10.5 g/dL, a leukocyte count of 9,000/mm3, a platelet count of 145,000/mm3, and a creatine kinase of 190 U/L. An ECG shows sinus tachycardia.",
    "A 55-year-old man comes to the emergency department because of a dry cough and severe chest pain beginning that morning. Two months ago, he was diagnosed with inferior wall myocardial infarction and was treated with stent implantation of the right coronary artery. He has a history of hypertension and hypercholesterolemia. His medications include aspirin, clopidogrel, atorvastatin, and enalapril. His temperature is 38.5\u00c2\u00b0C (101.3\u00b0F), pulse is 92/min, respirations are 22/min, and blood pressure is 130/80 mm Hg. Cardiac examination shows a high-pitched scratching sound best heard while sitting upright and during expiration. The remainder of the examination shows no abnormalities. An ECG shows diffuse ST elevations. Serum studies show a troponin I of 0.005 ng/mL (N < 0.01).",
    "A 30-year-old African American woman comes to the physician for the evaluation of a dry cough and chest discomfort for the past 3 days. During this period, the patient has had headaches, muscle aches, joint pain, fever, and chills. Ten days ago, she was hiking  with her family in Mississippi. The patient has asthma that is treated with an albuterol inhaler. Her mother has a lung disease treated with methotrexate. The patient has smoked one pack of cigarettes daily for the past 10 years. Her temperature is 38AH \u00b0C (100.4AH \u00b0F).  Physical examination shows slight wheezes throughout both lung Qields. Laboratory studies and urinalysis are positive for polysaccharide antigen. Bronchoalveolar lavage  using silver/PAS-staining shows macrophages Qilled with a dimorphic fungus with septate hyphae.""",
    """A 45-year-old male presents with abdominal pain, jaundice, and clay-colored stools for the past week. He reports no significant  medical history but mentions recent travel to Southeast Asia. Laboratory tests show elevated liver enzymes and bilirubin.""",
    """A 65-year-old female with a history of hypertension and long-term smoking suddenly experiences excruciating chest pain that radiates to her back.  Physical examination indicates diminished bilateral femoral pulses.""",
    """A middle-aged male presents with acute abdominal pain localized to the right lower quadrant. He mentions a recent overseas trip to a tropical region and recalls consuming untreated water and uncooked seafood. The patient complains of watery diarrhea with blood and mucus, accompanied by abdominal cramping and tenesmus. Physical examination shows signs of dehydration, and laboratory studies reveal leukocytosis with eosinophilia. What is the likely cause of his symptoms?"""
]

system = "Use step by step deduction to create a differential diagnosis considering the patient's symptoms, medical history, and test resultsand then use step by step deduction to determine the correct response."

llm = Llama(
    # path_to_model,
    chat_format="chatml",
    n_ctx=8192,
    n_gpu_layers=-1,
    verbose=False,
)

def get_output(context_index):
    context = questions[context_index] if context_index < len(questions) else ""
    question = "What are the potential diagnoses based on the symptoms and context? Identify the most probable diagnosis with a detailed rationale."
    input_text = " Contexts: " + context + " Question: " + question

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": input_text},
        ]
    )

    print("Context:")
    print(context)
    print(" Output:")
    print(output["choices"][0]["message"]["content"])


if len(sys.argv) > 1:
    context_index = int(sys.argv[1])
    get_output(context_index)
else:
    print("Please provide a valid context index as an argument.")
