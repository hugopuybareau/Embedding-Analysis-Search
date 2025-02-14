# Imports

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from embedding_models import EmbeddingModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI() # Initialize the API

model = EmbeddingModel(method='sbert') # Initialize the sbert model

texts = [
    "Artificial intelligence is transforming the financial sector with new predictive models.",
    "Stock markets experience volatility following the Federal Reserve's latest announcement.",
    "Tesla unveils a revolutionary battery with extended lifespan and record energy efficiency.",
    "Apple officially launches iPhone 16 with breakthrough AI-driven camera features.",
    "The job market in France sees significant improvements thanks to emerging tech jobs.",
    "AI and cybersecurity are reshaping modern data protection strategies.",
    "Google announces new AI model capable of generating 3D models from text descriptions.",
    "Microsoft's latest Azure update brings quantum computing capabilities to developers.",
    "SpaceX successfully tests its reusable rocket system for deep-space missions.",
    "Amazon introduces a drone delivery service capable of handling same-day shipping.",
    "Neuralink demonstrates successful brain-computer interface trials in human patients.",
    "OpenAI releases GPT-5 with improved contextual understanding and multimodal capabilities.",
    "Blockchain adoption accelerates in banking, providing secure transaction verification.",
    "Samsung reveals flexible OLED screens that could change the future of mobile devices.",
    "Quantum computing breakthrough: IBM unveils 1000-qubit processor for commercial use.",
    "Meta introduces a new AR headset with advanced spatial mapping and haptic feedback.",
    "Autonomous vehicles reach a new milestone with regulatory approval in multiple countries.",
    "NVIDIA announces AI-powered GPUs optimized for deep learning workloads.",
    "Google DeepMind's AlphaFold revolutionizes protein folding research for medical advances.",
    "Intel debuts its latest generation of AI-driven processors for high-performance computing.",
    "Apple M3 chip outperforms competition with 40% faster processing speeds and lower power consumption.",
    "Cybersecurity threats rise as hackers exploit AI-generated phishing emails.",
    "AI-driven robotics startups secure record funding to automate manufacturing processes.",
    "Microsoft Edge introduces AI-powered browsing assistant with enhanced user customization.",
    "Tesla’s Full Self-Driving software update achieves near-perfect urban navigation.",
    "European Union enforces strict AI regulations to ensure ethical development of machine learning systems.",
    "Amazon's Alexa gets major AI upgrades, allowing more natural conversations and task automation.",
    "China accelerates AI research with massive government funding for deep learning initiatives.",
    "Apple Vision Pro AR headset receives critical acclaim for immersive user experience.",
    "SpaceX Starship successfully completes orbital flight test with full reusability.",
    "Google Gemini AI outperforms human experts in data analysis and pattern recognition.",
    "AI-powered health diagnostics can now detect early-stage cancer with 98% accuracy.",
    "Cryptocurrency markets recover after regulatory clarity from global financial institutions.",
    "TikTok introduces AI-generated content moderation to detect harmful material faster.",
    "NASA’s Artemis mission to the Moon leverages AI navigation for autonomous landing.",
    "Meta launches AI-powered social media assistant capable of generating video summaries.",
    "IBM Watson AI now assists medical professionals in diagnosing rare diseases.",
    "Google Bard chatbot surpasses GPT models in contextual accuracy and factual recall.",
    "Waymo expands autonomous taxi services to new cities following successful trials.",
    "Apple Car project gains traction as self-driving prototype enters road testing.",
    "Amazon’s Just Walk Out technology reaches mainstream adoption in global retail stores.",
    "Scientists develop AI-driven weather forecasting models that outperform traditional systems.",
    "AI-powered music composition tools challenge the creativity of human composers.",
    "Deepfake technology raises concerns as governments struggle to regulate synthetic media.",
    "AI-driven fraud detection tools help banks prevent billions in financial losses.",
    "Intel announces new AI accelerator chips designed for data centers and cloud computing.",
    "Microsoft partners with OpenAI to integrate ChatGPT-like AI into enterprise software.",
    "YouTube experiments with AI-generated video summaries for faster content browsing.",
    "Elon Musk's Neuralink receives FDA approval for large-scale human trials.",
    "AI-generated art gains recognition in global exhibitions and museum collections.",
    "Google Search gets AI-powered upgrades for personalized and context-aware results.",
    "Meta introduces AI-driven content moderation to detect misinformation in real time.",
    "AI-generated text-to-video tools gain popularity among content creators.",
    "ChatGPT API enables developers to build powerful conversational AI assistants.",
    "Tesla's Optimus humanoid robot enters beta testing for household assistance.",
    "Amazon Web Services introduces AI-based cloud security features.",
    "AI-powered medical research accelerates drug discovery for rare diseases.",
    "SpaceX plans AI-managed satellite networks for optimized global internet coverage.",
    "Microsoft’s Copilot AI assistant now available in Office Suite for business users.",
    "Google’s AI-powered translation system surpasses human-level accuracy in multiple languages.",
    "Autonomous delivery robots begin large-scale deployment in urban areas.",
    "AI-powered handwriting recognition improves accessibility for visually impaired users.",
    "Apple Watch gains new AI-driven health monitoring features for early disease detection.",
    "AI-powered gaming engines revolutionize real-time world generation and NPC interactions.",
    "Google Assistant gets AI-powered upgrades for deeper context awareness and task automation.",
    "AI-generated deepfake scams increase, prompting new cybersecurity measures.",
    "Drones powered by AI navigate disaster zones to assist first responders.",
    "AI in agriculture improves crop yield predictions and automates precision farming.",
    "AI-powered traffic management systems reduce congestion in major cities.",
    "Cybersecurity firms adopt AI to detect threats before they materialize.",
    "AI-designed materials pave the way for ultra-efficient solar panels.",
    "AI-generated educational content personalizes learning for students worldwide.",
    "Smart home devices gain enhanced AI capabilities for predictive automation.",
    "AI-driven legal analysis tools assist lawyers in case preparation and research.",
    "Facebook's AI moderation system detects and removes hate speech with 99% accuracy.",
    "Tesla’s new AI-powered supercomputer enhances autonomous driving capabilities.",
    "AI-driven fintech solutions disrupt traditional banking models.",
    "Augmented reality shopping experiences powered by AI transform e-commerce.",
    "AI-generated voice synthesis reaches near-perfect human-like speech patterns.",
    "Google Photos introduces AI-driven image enhancement for low-light photography.",
    "Autonomous drones begin delivering medical supplies in remote areas.",
    "AI-powered language translation technology removes barriers in international business.",
    "Blockchain-integrated AI systems improve transparency in digital transactions.",
    "AI-powered recommendation systems redefine content consumption across streaming platforms.",
    "AI-driven personal finance assistants help users optimize savings and investments.",
    "Elon Musk hints at AI-powered humanoid robots for industrial applications.",
    "AI-assisted journalism tools generate news summaries in real time.",
    "AI-driven video game NPCs exhibit human-like behavior in open-world environments.",
    "AI-powered voice recognition systems achieve 99% accuracy across multiple languages.",
    "AI-driven stock trading algorithms outperform human analysts in market predictions.",
    "AI-powered sentiment analysis transforms marketing strategies for businesses.",
    "Autonomous ships begin commercial trials for AI-powered cargo transport.",
    "Quantum AI research accelerates breakthroughs in secure cryptographic systems.",
]


model.fit(texts) # Train and Index the texts
model.index_texts(texts)

class QueryRequest(BaseModel): # Pydantic model for the query
    query: str
    how_much_results: int = 3

class DocumentRequest(BaseModel): 
    texts: List[str]

@app.get("/") #GET route
def home():
    return {"message": "API running"}

@app.post("/search/") # Post endpoint because user sends input data
def search(request: QueryRequest):
    results = model.search_similar(request.query, request.how_much_results)
    return {"query": request.query, "results": [{"document": doc, "score": f"{float(score):.2f}, "} for doc, score in results]}

@app.post("/index/") # Post endpoint because user sends data again
def index_new_texts(request: DocumentRequest):
    model.index_texts(request.texts)
    return {"message": f"Indexed {len(request.texts)} new texts."}

# Enable CORS for all domains (or restrict to your frontend) to dodge 405 method not allowed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods 
    allow_headers=["*"],
)

