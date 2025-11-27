# final.py - Streamlit-ready inference only

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import random
from collections import Counter
import matplotlib.pyplot as plt

# =========================
# 1️⃣ Load tokenizer + model
# =========================
device = torch.device("cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

num_labels = 28
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels
)
model.load_state_dict(torch.load("model_epoch2.pt", map_location=device))
model.to(device)
model.eval()

# =========================
# 2️⃣ Define emotions & responses
# =========================

response_templates = {
    "joy": [
        "That's wonderful to hear! 😄 What’s making you feel so good today?",
        "I love your positive energy! ✨ Tell me more about it.",
        "Happiness looks good on you 😍 What’s the reason?",
        "Yay! 🎉 I’m happy that you’re happy."
    ],
    "sadness": [
        "I'm sorry you’re feeling this way 💙 Want to talk about what happened?",
        "That sounds tough 😔. I’m here to listen if you’d like to share.",
        "I can imagine this must be hard 💭. You're not alone in this.",
        "Sending you a virtual hug 🤗. Want me to distract you with something positive?"
    ],
    "anger": [
        "I can sense some frustration 😠. Do you want to let it out?",
        "That sounds upsetting 💢. What’s bothering you the most?",
        "It’s okay to feel angry. Sometimes venting helps—want to try?",
        "I hear you 🔥. Let’s talk through what made you feel this way."
    ],
    "fear": [
        "That sounds scary 😟. Do you want to tell me more about it?",
        "I get it, fears can be overwhelming 💭. You're safe here.",
        "It must be unsettling 😣. What’s making you feel uneasy?",
        "I’m here with you 🤝. Want to break it down together?"
    ],
    "love": [
        "Aww, that’s heartwarming ❤️ Tell me more!",
        "That’s really sweet 🥰. Who or what makes you feel this way?",
        "Love makes life brighter 💕. I’d love to hear the story.",
        "That put a smile on my face 😊. Want to share more details?"
    ],
    "surprise": [
        "Wow, that sounds unexpected! 😲 What happened?",
        "Oh really?! 🤯 Tell me more about that.",
        "That must’ve been quite a shock 😮. How are you feeling about it?",
        "Oh wow! 🌟 I didn’t see that coming either!"
    ],
    "neutral": [
        "Got it 👍. I’m listening—go on.",
        "Okay 🙂 I’m following along, tell me more.",
        "I hear you 👂. What’s next?",
        "Alright, thanks for sharing. Do you want to expand on that?"
    ],
    "annoyance": [
        "Sounds like something’s bothering you 😒. Want to talk about it?",
        "I get it, little things can be irritating 🌀. What happened?",
        "That must feel annoying 😑. How are you handling it?",
        "I hear your frustration 🫤. Want to unpack it?"
    ],
    "boredom": [
        "Feeling bored, huh? 😐 Want me to share something fun?",
        "Sometimes boredom leads to creativity 🎨. Want an idea?",
        "I get it—slow days can drag 🕒. What usually cheers you up?",
        "Want me to suggest a quick activity to beat the boredom?"
    ],
    "disgust": [
        "Yikes, that must’ve felt unpleasant 🤢. What happened?",
        "I hear your disgust. Want to vent about it?",
        "That reaction makes sense 💭. Do you want to explain?",
        "Sometimes things can really put us off 😖. What caused it?"
    ],
    "guilt": [
        "It sounds like you’re blaming yourself 😔. Want to talk it through?",
        "Guilt can be heavy 💭. Remember, mistakes happen to everyone.",
        "I hear you—what’s making you feel guilty?",
        "Being kind to yourself matters 🤍. Want me to remind you of that?"
    ],
    "shame": [
        "That sounds like a tough feeling 😞. Do you want to share more?",
        "Shame can weigh a lot 💭. Remember, you deserve compassion.",
        "You’re not defined by mistakes 🤍. Want to talk about it?",
        "I hear you—what’s making you feel ashamed?"
    ],
    "pride": [
        "That’s amazing! 🌟 You should be proud.",
        "I love hearing your accomplishments 🎉. Tell me more!",
        "That’s worth celebrating 🥂. What are you proud of?",
        "Yes! Own that success 💪. You earned it."
    ],
    "relief": [
        "Phew 😌 That must feel better.",
        "I’m glad things eased up for you 🌸.",
        "Relief is such a calming feeling 💆. Want to share what happened?",
        "That’s wonderful! 🎉 What lifted the weight off your shoulders?"
    ],
    "trust": [
        "That’s a big step 🤝. Who or what do you trust?",
        "Trust is powerful 💭. It means a lot that you feel it.",
        "I hear you—building trust takes time 🕰️.",
        "That’s heartwarming ❤️. Want to share more?"
    ],
    "anticipation": [
        "Ooo exciting! 👀 What are you waiting for?",
        "Anticipation can feel thrilling ⚡. Want to tell me about it?",
        "Sounds like you’re looking forward to something 🎉.",
        "Can’t wait with you! 🙌 What’s coming up?"
    ],
    "curiosity": [
        "Ooo, that’s an interesting thought 🤔. What’s on your mind?",
        "I love curious questions 🧠. Tell me more!",
        "Curiosity sparks discovery 🔎. Want to explore together?",
        "What got you curious about this?"
    ],
    "confusion": [
        "Hmm 🤨 sounds confusing. Want me to help untangle it?",
        "I get it, things can be unclear sometimes 💭.",
        "That does sound puzzling 🧩. Tell me more.",
        "Let’s sort it out together 👥. What’s confusing you?"
    ],
    "hope": [
        "That’s uplifting ☀️. What are you hopeful about?",
        "Hope gives strength 💪. Tell me more!",
        "I love your optimism ✨. What makes you feel this way?",
        "Hope keeps us going 🌱. Want to share yours?"
    ],
    "disappointment": [
        "I’m sorry it didn’t go as expected 😔.",
        "That must feel discouraging 💭. Want to talk about it?",
        "Disappointment can be hard 😕. What happened?",
        "I hear you. Do you want to share what let you down?"
    ],
    "embarrassment": [
        "Oof, that sounds awkward 😅. Want to laugh about it?",
        "I get it—embarrassing moments happen to everyone 💭.",
        "That must’ve felt uncomfortable 😳. What happened?",
        "Don’t worry, you’re not alone in this 🤍."
    ],
    "gratitude": [
        "That’s beautiful 🙏. What are you grateful for?",
        "Gratitude warms the heart 💕. Tell me more!",
        "I love that mindset 🌟. What made you thankful?",
        "That’s inspiring ✨. Share the moment with me?"
    ],
    "loneliness": [
        "I hear you 💙. Loneliness can be tough.",
        "It must feel isolating 😔. Want to talk about it?",
        "You’re not alone—I’m here for you 🤝.",
        "Would you like me to suggest ways to connect with others?"
    ],
    "nostalgia": [
        "Ah, memories 🌙. What made you think of that?",
        "Nostalgia can be sweet 💭. Want to share the story?",
        "That must’ve been a special time ✨.",
        "I love hearing about fond memories 🌸."
    ],
    "excitement": [
        "Yay! 🎉 What’s got you so excited?",
        "I love your energy ⚡. Tell me more!",
        "That sounds amazing 😍. What’s happening?",
        "Woohoo! 🎊 I’m excited with you!"
    ],
    "insecurity": [
        "I hear you 💙. Insecurities can be tough.",
        "Remember—you have value exactly as you are 🌟.",
        "Want to share what’s making you feel insecure?",
        "You deserve kindness, even from yourself 🤍."
    ],
    "envy": [
        "That sounds like envy 😕. Want to unpack it?",
        "It’s okay to feel that way sometimes 💭.",
        "What do you wish you had for yourself?",
        "Remember, your journey matters 🌱."
    ],
    "admiration": [
        "Wow, that’s inspiring 🌟. Who do you admire?",
        "That’s a wonderful quality 💕. Tell me more!",
        "I love hearing about admiration—it spreads positivity ✨.",
        "That’s awesome 🙌. What inspires you most?"
    ]
}

import random

response_templates.update({
    "approval": [
        "I see, that’s a good choice 👍",
        "Sounds like the right move!",
        "I totally agree with you!",
        "That’s really smart!",
        "I support that decision!",
        "Yes, that makes sense!",
        "Absolutely, well done!",
        "You’ve got a point there!",
        "I can see why you approve!",
        "Great thinking!"
    ],
    "disgust": [
        "Yikes, that doesn’t sound good 🤢",
        "I get why that bothers you",
        "Ugh, that’s unpleasant",
        "That must feel gross 😖",
        "I understand your disgust",
        "Hmm, not very nice indeed",
        "That’s really off-putting",
        "I see why that makes you uncomfortable",
        "Yuck! I get it",
        "That seems awful"
    ],
    "disapproval": [
        "I see why you disapprove",
        "That’s understandable",
        "Hmm, not ideal",
        "I get your concern",
        "I hear your disagreement",
        "Yes, that doesn’t seem right",
        "I understand your objection",
        "I can see why you feel that way",
        "Not the best choice, I agree",
        "Your point is clear"
    ],
    "remorse": [
        "I understand you feel regretful",
        "It’s okay, everyone makes mistakes",
        "I get why you feel remorse",
        "Don’t be too hard on yourself",
        "I hear your apology",
        "It’s normal to feel that way",
        "I understand, you feel sorry",
        "It’s good to acknowledge it",
        "Feeling remorse shows empathy",
        "I know it weighs on you"
    ],
    "curiosity": [
        "That’s interesting, tell me more",
        "I’m curious too!",
        "What makes you wonder about that?",
        "I love your curiosity",
        "That’s a good question!",
        "I’d like to know as well",
        "Keep exploring, that’s great",
        "I’m intrigued by your thought",
        "Tell me more about that curiosity",
        "Fascinating!"
    ],
    "relief": [
        "I’m glad that worked out 😌",
        "That must be a weight off your shoulders",
        "I understand your relief",
        "Glad you feel better now",
        "That’s good to hear",
        "Phew, that’s a relief indeed",
        "I see, you feel reassured",
        "That’s comforting to know",
        "It’s nice to relax now",
        "I’m happy it’s resolved"
    ],
    "admiration": [
        "Wow, that’s impressive 😮",
        "I admire that too!",
        "That’s really inspiring",
        "I can see why you admire that",
        "Amazing work!",
        "I respect that greatly",
        "That’s quite admirable",
        "Truly remarkable",
        "I’m impressed by that",
        "You’ve got great taste"
    ],
    "realization": [
        "Ah, I see what you mean now",
        "That makes sense",
        "I understand now",
        "Interesting realization!",
        "I get it, thanks for explaining",
        "That’s a good insight",
        "Ah, now it’s clear",
        "I understand your point",
        "That’s enlightening",
        "I see, good observation"
    ],
    "optimism": [
        "That’s a hopeful thought 🌟",
        "I like your positive outlook",
        "Stay optimistic, good things ahead",
        "That’s encouraging",
        "I see the bright side too",
        "Yes! Things will improve",
        "Keep believing in the best",
        "That’s a confident perspective",
        "Hope is always good",
        "Your optimism is inspiring"
    ],
    "amusement": [
        "Haha, that’s funny 😄",
        "I see why that made you laugh",
        "That’s quite amusing",
        "I can’t stop smiling 😆",
        "You made me chuckle too",
        "That’s hilarious!",
        "I love that sense of humor",
        "Too funny! 😂",
        "You have a good sense of fun",
        "That made me laugh"
    ],
    "gratitude": [
        "I appreciate that 🙏",
        "That’s very kind of you",
        "Thanks for sharing that",
        "I’m grateful too",
        "I feel thankful",
        "Much appreciated",
        "Thank you for telling me",
        "I’m glad for that",
        "I really value that",
        "I’m grateful for your words"
    ],
    "caring": [
        "That’s really thoughtful of you",
        "I see you care about this",
        "That’s so kind 💖",
        "Your care is evident",
        "You’re very considerate",
        "I appreciate your concern",
        "That’s very nurturing",
        "You really care, I see",
        "It’s touching how much you care",
        "I understand your compassion"
    ],
    "confusion": [
        "I get that you’re confused 🤔",
        "Hmm, that seems unclear",
        "I understand your puzzlement",
        "It’s okay to be confused",
        "Let’s figure it out together",
        "I see, that’s tricky",
        "I get why you feel uncertain",
        "Hmm, let’s clarify",
        "I understand your doubt",
        "That is confusing indeed"
    ],
    "excitement": [
        "That’s thrilling! 😆",
        "I’m excited for you!",
        "Wow, that’s exciting news!",
        "I love your enthusiasm",
        "That must feel amazing",
        "So pumped for you!",
        "Can’t wait to hear more!",
        "Yay! That’s energizing",
        "I see your excitement!",
        "Keep that energy up!"
    ],
    "embarrassment": [
        "I understand that’s awkward 😳",
        "Don’t worry, it happens",
        "I get why you feel embarrassed",
        "It’s okay, we all do that",
        "I hear you, that’s uncomfortable",
        "I know that feeling",
        "You’re not alone in that",
        "That must be embarrassing",
        "I see why that’s awkward",
        "It’s alright, no worries"
    ],
    "pride": [
        "That’s worth celebrating 🥂",
        "You should be proud of yourself!",
        "Great job! 💪",
        "I admire your achievement",
        "Well done!",
        "That’s an accomplishment",
        "I see your pride!",
        "You’ve earned this recognition",
        "Congrats on your success!",
        "That’s fantastic!"
    ],
    "nervousness": [
        "It’s okay to be nervous 😬",
        "I understand your anxiety",
        "Take a deep breath, I’m here",
        "I get why you’re uneasy",
        "That’s a bit stressful, isn’t it?",
        "I see your worry",
        "You’ll get through this",
        "It’s normal to feel anxious",
        "I understand the tension",
        "Stay calm, you’ve got this"
    ],
    "desire": [
        "I see what you’re longing for",
        "That sounds like a strong wish",
        "I understand your desire",
        "I hope you get it soon",
        "That’s a meaningful goal",
        "You really want that, I get it",
        "I understand your yearning",
        "Keep striving for it",
        "Your desire is clear",
        "That’s something you care about"
    ],
    "grief": [
        "I’m sorry for your loss 💔",
        "That must be really hard",
        "I understand your sorrow",
        "Take your time to grieve",
        "I hear your pain",
        "It’s okay to feel sad",
        "I feel for you deeply",
        "I’m here if you want to talk",
        "Grieving takes time",
        "Sending you comfort"
    ]
})

# For emotions without pre-defined responses, you can fallback
emotions_28 = [
    'approval','joy','disgust','disapproval','remorse','curiosity','relief',
    'admiration','realization','anger','optimism','amusement','neutral',
    'gratitude','annoyance','sadness','disappointment','caring','confusion',
    'love','excitement','surprise','embarrassment','fear','pride',
    'nervousness','desire','grief'
]

for e in emotions_28:
    if e not in response_templates:
        response_templates[e] = ["Hmm, I’m listening 👂"]

# =========================
# 3️⃣ Prediction functions
# =========================
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return emotions_28[predicted_class_id]

def chatbot_response(user_input):
    # Critical keyword check
    critical_keywords = ["suicide", "self harm", "kill myself", "end my life", "hopeless"]
    if any(kw in user_input.lower() for kw in critical_keywords):
        return "CRITICAL", (
            "⚠️ It seems you might be in serious trouble. "
            "Please contact the helpline immediately: 18002701008"
        )
    
    emotion = predict_emotion(user_input)
    response = random.choice(response_templates.get(emotion, ["Hmm, I’m listening 👂"]))
    return emotion, response

def predict_conditions(conversation_log):
    emotions = [entry["emotion"] for entry in conversation_log]
    counts = Counter(emotions)
    total = len(emotions)
    conditions = []

    if total == 0:
        return ["No data to analyze"]

    if counts.get("sadness", 0)/total > 0.3:
        conditions.append("Depression")
    if counts.get("fear", 0)/total > 0.2 or counts.get("nervousness", 0)/total > 0.2:
        conditions.append("Anxiety")
    if counts.get("anger", 0)/total > 0.2 or counts.get("annoyance", 0)/total > 0.2:
        conditions.append("Stress / Anger Management Concerns")
    if counts.get("grief", 0)/total > 0.1:
        conditions.append("Grief / Loss-related Stress")
    if counts.get("joy", 0)/total > 0.5:
        conditions.append("Overall Positive Mood")

    # Critical keywords override
    critical_keywords = ["suicide", "self harm", "kill myself", "end my life", "hopeless"]
    for entry in conversation_log:
        text = entry["text"].lower()
        if any(word in text for word in critical_keywords):
            conditions.append("Immediate Crisis - Seek Help")
            break

    if not conditions:
        conditions.append("No clear indication")

    return conditions
def generate_report(conversation_log):
    emotions = [entry["emotion"] for entry in conversation_log]
    counts = Counter(emotions)
    total = len(emotions)

    report = f"Total messages: {total}\n"
    report += "Emotion distribution:\n"
    for e, c in counts.items():
        report += f"  {e}: {c} ({c/total*100:.1f}%)\n"

    # Check for negative emotions
    risk = ["sadness","anger","fear","annoyance","grief"]
    flagged = {e: counts[e] for e in risk if e in counts}
    if flagged:
        report += "\n⚠️ Potential concerns detected:\n"
        for e, c in flagged.items():
            report += f"  - {e}: {c} times\n"
        report += "Consider consulting a mental health professional if negative emotions persist.\n"
    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values(), color='skyblue')
    ax.set_ylabel('Count')
    ax.set_xlabel('Emotions')
    ax.set_title('Emotion Distribution in Conversation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return report, fig
