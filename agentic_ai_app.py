from flask import Flask, render_template, request, jsonify, redirect, session
import torch
import re
from sentence_transformers import util
import pickle
import tensorflow as tf
from keras.models import load_model
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
# from langchain.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
import os
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

app = Flask(__name__)
app.secret_key = os.urandom(24) 
genai.configure(api_key="AIzaSyCRw5CDXp7ad6U9Uwjac-sc_Xcd7gLNaso")
llm = GoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key="AIzaSyCRw5CDXp7ad6U9Uwjac-sc_Xcd7gLNaso")

service_images = {
    'Loan': '/static/loan.jpeg',
    'Training': '/static/traning.jpg',
    'Subsidy': '/static/subsidy.jpg',
    'Market Access': '/static/market access.jpg',
    'Soil Testing': '/static/soil testing.jpg',
    'Crop Selection Advisory': '/static/crop selection adivisory.jpg',
    'Weather Alerts': '/static/weather alert.jpg',
    'Irrigation Plans': '/static/irrigations.jpg',
    'Organic Farming Support': '/static/organic farming support.jpg',
    'Precision Farming': '/static/precision farming.jpg',
    'Crop-Specific Training': '/static/organic farming vegitable.jpg',
    'Wheat Monitoring': '/static/wheat monitoring.jpg',
    'Corn Disease Detection': '/static/corn diseases detection.jpg',
    'Rice Water Management': '/static/rice water management.jpg',
    'Vegetable Organic Farming': '/static/vegitable organic farming.jpg',
    'Water Analysis Facility': '/static/Water Analysis Facility.jpg',
    'Tractor Booking Facility':'/static/Tractor Booking Facility.jpg',
    'Seed Selection Advisory':'/static/Seed Selection Advisory.jpg',
    'Fertilizer Recommendation': '/static/Fertilizer Recommendation.jpg',
    'Pest and Disease Control': '/static/Pest and Disease Control.jpg',
    'Weather Forecasting': '/static/Weather Forecasting.jpg',
    'Government Scheme Assistance':'/static/Government Scheme.webp',
    'Rental Equipment Facility': '/static/Rental Equipment Facility.jpg',
    'Insurance Service':'/static/Insurance Service.jpg'
      
}

# Load necessary data
embeddings = pickle.load(open('service_embeddings_new.pkl', 'rb'))
if isinstance(embeddings, dict):
    embeddings = torch.tensor(list(embeddings.values()))
services = pickle.load(open('services_new.pkl', 'rb'))
rec_model = pickle.load(open('rec_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# loading of model neural network 
# query_type_model = tf.keras.models.load_model('C:\\Users\\DELL\\small recommendation system\\neural_network_model_for_QueryType.h5')  # Load your query type model
# query_type_model = load_model("optimized_model.h5") 
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()
def services_ranking(scores_list):
    # Get all services with their similarity scores
    all_services_with_scores = [
        (list(services.keys())[idx], scores_list[idx]*100) for idx in range(len(scores_list))
    ]
    all_services_with_scores.sort(key=lambda x: x[1], reverse=True)
    # print(all_services_with_scores)


# Recommendation function
def recommendation(farmer_issues):
    
    cosine_scores = util.cos_sim(rec_model.encode(farmer_issues, convert_to_tensor=True), embeddings)
    scores_list = cosine_scores[0].tolist()
    # services_ranking(scores_list)
    print(scores_list)
    top_results = torch.topk(cosine_scores, k=6) # get top 3 recommendation

    indices = top_results.indices[0].tolist() # extract the indices of top result
    top_scores = top_results.values[0].tolist()
    recommended_services = [
        list(services.keys())[idx] for idx in indices
        ]
    print("recommended service",recommended_services)
    # print(query_type)

    #services releated to querytype{
    # releated_service_to_queryType =[]
    # for service, description in services.items():
    #     description_embedding = rec_model.encode(description, convert_to_tensor =True)
    #     query_embbeding = rec_model.encode(query_type, convert_to_tensor =True)
    #     similarity_score = util.cos_sim(query_embbeding ,description_embedding )
    #     if similarity_score > 0.3:  # Threshold for similarity
    #         releated_service_to_queryType.append(service)
    #}

    # the main one ->
    # similarity_score= util.cos_sim(rec_model.encode(query_type, convert_to_tensor=True), embeddings)
    # similarity_score_list = similarity_score[0].tolist()
    # services_ranking(similarity_score_list)
    # # print(similarity_score)
    # print(similarity_score_list)
    # # print(similarity_score[0].tolist())
    # top_results = torch.topk(similarity_score, k=6)
    # indices1 = top_results.indices[0].tolist()
    # releated_service_to_queryType=[
    #     list(services.keys())[idx1] for idx1 in indices1
    # ]

    # print(releated_service_to_queryType)

    # all_recommendations = list(set(recommended_services + releated_service_to_queryType))
    # print(all_recommendations)

    filtered_recommendations = []

# Filter recommendations from farmer_issues
    for idx, score in zip(indices, top_scores):
        # normalized_score = normalize_score(score)
        if score >= 0.42:
            filtered_recommendations.append(list(services.keys())[idx])

    # Filter recommendations from query_type
    # for idx1, score in zip(indices1, similarity_score[0].tolist()):
    #     # normalized_score = normalize_score(score)
    #     if score >= 0.35:
    #         filtered_recommendations.append(list(services.keys())[idx1])

    # Remove duplicates
    # filtered_recommendations = list(filtered_recommendations)
    print("filter recommendation",filtered_recommendations)
    return filtered_recommendations


#function for query_type prediction 
def predict_query_type(user_input):

    user_input_cleaned = user_input.lower().replace('[^\w\s]', '')
    user_input_vectorized = vectorizer.transform([user_input_cleaned]).toarray()
    # user_input_vectorized = tf.sparse.reorder(user_input_vectorized)
    query_type = query_type_model.predict(user_input_vectorized)
    predicted_index = np.argmax(query_type, axis=1)

    # Decode the predicted index to get the query type
    predicted_query_type = label_encoder.inverse_transform(predicted_index)

    print(predicted_query_type)
    return predicted_query_type[0].strip()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # query_cleaned = user_input.lower().replace('[^\w\s]', '')  # Normalize text

    # query_vectorized = vectorizer.transform([query_cleaned]).toarray()
    # query_vectorized = query_vectorized.astype(np.float32)
    # interpreter.set_tensor(input_details[0]['index'], query_vectorized)
    # interpreter.invoke()
    # prediction = interpreter.get_tensor(output_details[0]['index'])
    # predicted_index = np.argmax(prediction, axis=1)
    # print(predicted_index)
    # predicted_query_type = label_encoder.inverse_transform(predicted_index)

    # return predicted_query_type[0]
    


# Follow-up question generator using Gemini
def generate_followup_question(user_query):
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    prompt = f"""Act as an agricultural expert. Given this user query: "{user_query}", 
    ask one concise follow-up question to better understand their problem. 
    Keep it natural and focused on farming-related aspects."""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip('"').strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Could you please provide more details about your agricultural issue?"

# AI Agent to provide structured answers when no recommendation is found
search_tool = DuckDuckGoSearchRun()
def ai_agent_answer(user_query):
    """
    Fetches structured information using a combination of Gemini and DuckDuckGo search.
    """
    
    search_tool = DuckDuckGoSearchRun()

    agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
    )

    try:
        # 1. Initial Gemini Response
        # initial_response = llm.invoke(f"""
        # Please provide a **detailed and well-structured** response to the following question:

        # {user_query}    

        # Your response should include explanations, examples, and relevant details where necessary.
        #     """)
        initial_response=llm.invoke(f"Think you are agriculture experst.Please provide a concise answer to the following question: {user_query} in 100 to 200 words by which farmere can get initial idea.")
        print("initial response :",initial_response)

        # 2. DuckDuckGo Search for Further Information
        search_results = agent.run(user_query)
        print("search result :",search_results)
        # 3. Combine Gemini and Search Results
        # Ask Gemini to refine its answer based on the search results
        final_prompt = f"""

        You initially provided this response: {initial_response}.

        Here are some search results related to the query: {user_query}:
        {search_results}
        Please analyze both initial response and search resuult and provide a best result which for user query : {user_query} by thinking that you are an agriculture experst 
        Response should be accurate and comprehensive answer and it should be in max 300 wrods
       
        """
        # final_prompt = f"""
        #     Analyze the following information carefully and provide an accurate, well-structured, and easy-to-understand response to the user's query.

        #     **Initial AI Response (Based on Knowledge & Reasoning)**:  
        #     {initial_response}

        #     **Additional Search Results (External Sources & Relevant Information)**:  
        #     {search_results}

        #     **Task:**  
        #     - Compare both responses, analyze key insights, and identify the most reliable and relevant details.  
        #     - Correct any factual inconsistencies or missing points.  
        #     - Ensure clarity, coherence, and simplicity so that a farmer can easily understand the answer.  
        #     - Use bullet points or short paragraphs if needed to enhance readability.

        #      Provide the best possible response to the user's query:  
        #     "{user_query}"
        #     """


        final_response = llm.invoke(final_prompt)

        print("suggested_action :",final_response)
        print
        return {
            "structured_response": {
                "query": user_query,
                "gemini_initial_response": initial_response,
                "search_results": search_results,
                "suggested_action": final_response
            }
        }
    except Exception as e:
        print(f"AI Agent Error: {e}")
        return {"structured_response": "No relevant information found."}
# def ai_agent_answer(user_query):
#     """Fetch structured information when no recommendations are found."""
#     agent = initialize_agent(
#         tools=[search_tool],
#         llm=llm,
#         agent="zero-shot-react-description",
#         verbose=True
#     )

#     try:
#         response = agent.run(user_query)
#         return {
#             "structured_response": {
#                 "query": user_query,
#                 "suggested_action": response
#             }
#         }
#     except Exception as e:
#         print(f"AI Agent Error: {e}")
#         return {"structured_response": "No relevant information found."}
    
# def scheme_display_agent(user_query):
#     try:
#         print("🔹 Pricing Agent Activated")  # Debugging Log
#     # Step 1: Use DuckDuckGo search to find schemes
#         # search_tool = DuckDuckGoSearchRun()
#         search_query = f"Latest government schemes for farmers in India related to {user_query}"
#         search_results = search_tool.run(search_query)
#         # search_result = search_tool.run(f"Latest government schemes for farmers in India related to {user_query}")
#         print(f"🔍 Searching for: {search_query}")
#     # Step 2: Use Gemini LLM to refine and structure the results
#         model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
#         prompt = (
#             f"You are an AI assistant for farmers. Based on the user's query: '{user_query}', "
#             "analyze the following search results and extract **only the top 3 most relevant schemes**.\n\n"
#             "Each scheme should include:\n"
#             "- Name\n"
#             "- Short Description\n"
#             "- Eligibility\n"
#             "- Financial Support\n\n"
#             f"Here is the raw data:\n{search_results}"
#         )

#         refined_schemes = model.generate_content(prompt)
#         print(f" Response from Gemini: {refined_schemes.text.strip()}")

#         return {"filtered_schemes": refined_schemes.text.strip()}

#     except Exception as e:
#         print(f"Pricing Agent Error: {e}")
#         return {"error": "Could not retrieve relevant pricing details."}
    
import json

def scheme_display_agent(user_query):
    try:
        print("Scheme agent activated")  # Debugging Log
        
        # Step 1: Use DuckDuckGo search to find schemes
        search_query = f"Latest government schemes for farmers in India related to {user_query}"
        search_results = search_tool.run(search_query)  # Assuming search_tool is defined elsewhere
        print(f" Searching for: {search_query}")
        
        # Step 2: Use Gemini LLM to refine and structure the results
        model = genai.GenerativeModel('gemini-2.0-flash-lite')

        prompt = (
            f"You are an AI assistant for farmers. Based on the user's query: '{user_query}', "
            "analyze the following search results and extract **only the top 3 most relevant schemes**.\n\n"
            "Return the response in **valid JSON format** with the following structure:\n\n"
            "{\n"
            '    "schemes": [\n'
            '        {\n'
            '            "name": "Scheme Name",\n'
            '            "description": "Short Description",\n'
            '            "eligibility": "Eligibility Criteria",\n'
            '            "financial_support": "Financial Support Details"\n'
            '        },\n'
            '        ...\n'
            '    ]\n'
            "}\n\n"
            f"Here is the raw data:\n{search_results}"
        )

        refined_schemes = model.generate_content(prompt)
        print(refined_schemes.text.strip())
        json_match = re.search(r'\{.*\}', refined_schemes.text.strip(), re.DOTALL)
        # Step 3: Convert response to structured JSON
        if json_match:
            json_text = json_match.group()  # Extract JSON portion
            try:
                structured_data = json.loads(json_text)  # Convert JSON string to Python dictionary
                return structured_data  # Return structured JSON data
            except json.JSONDecodeError:
                print("❌ Error: Extracted text is not valid JSON.")
                return {"error": "Invalid response format from AI agent."}
        else:
            print("❌ Error: No JSON found in response.")
            return {"error": "Invalid response format from AI agent."}
    except Exception as e:
        print(f"Pricing Agent Error: {e}")
        return {"error": "Could not retrieve relevant pricing details."}


@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/tryservice')
def tryservice():
    return render_template('index_new.html')

@app.route('/recommendation', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        farmer_issues = request.form.get('farmer_issues', '').strip()
        followup_answer = request.form.get('followup_answer', '').strip()
        previous_query = request.form.get('previous_query', '')

        # Use follow-up answer if provided
        if followup_answer:
            farmer_issues = followup_answer

        if not farmer_issues:
            return render_template('index_new.html', error="Please enter farmer issues.")

        # Fetch session counter (to track follow-up attempts)
        followup_count = session.get('followup_count', 0)

        # Predict query type
        # query_type = predict_query_type(farmer_issues)
        recommendations = recommendation(farmer_issues)

        # If recommendations found, display them
        if recommendations:
            session.pop('followup_count', None)  # Reset follow-up count
            return render_template('result4.html', recommendations=recommendations, services=services, service_images=service_images)

        # Limit follow-up attempts to 1 (or any desired threshold)
        if followup_count < 1:
            followup_question = generate_followup_question(farmer_issues)
            if followup_question:
                session['followup_count'] = followup_count + 1  # Increment follow-up count
                return render_template('index_new.html', followup_question=followup_question, previous_query=farmer_issues)

        # If no relevant results even after follow-up, use AI Agents
        structured_response = ai_agent_answer(farmer_issues)
        scheme_display = scheme_display_agent(farmer_issues)
        print(scheme_display )
        print(structured_response)
   
        # Reset follow-up count after AI agents step in
        session.pop('followup_count', None)
        
        return render_template('result4.html', structured_response=structured_response, pricing_data=scheme_display)

    return render_template('index_new.html')



# @app.route('/recommendation', methods=['POST'])
# def recommend():
#     """
#     Handles the recommendation request from the user.
#     If no recommendations are found, a follow-up question is generated.
#     If a follow-up has already been provided but no results are found, AI Agent & Pricing Agent are activated.
#     """
#     farmer_issues = request.form.get('farmer_issues', '').strip()
#     followup_answer = request.form.get('followup', '').strip()

#     # Use follow-up answer if provided
#     if followup_answer:
#         farmer_issues = followup_answer

#     if not farmer_issues:
#         return render_template('index_new.html', error="Please enter farmer issues.")

#     query_type = predict_query_type(farmer_issues)

#     # Get recommendations
#     recommendations = recommendation(farmer_issues, query_type)

#     # If recommendations exist, render them
#     print(recommendations)
#     if recommendations:
#         return render_template('result2.html', recommendations=recommendations, services=services, service_images=service_images)

#     # If no recommendations, generate a follow-up question
#     if not followup_answer:
#         followup_question = generate_followup_question(farmer_issues)
#         return jsonify({"followup": followup_question})

#     # If follow-up was already asked and still no result, activate AI Agent & Pricing Agent
#     structured_response = ai_agent_answer(farmer_issues)
#     pricing_data = scheme_display_agent(farmer_issues)

#     return jsonify({
#         "structured_response": structured_response,
#         "pricing_data": pricing_data
#     })

# @app.route('/recommendation', methods=['GET', 'POST'])
# def recommend():
#     if request.method == 'POST':

#         farmer_issues = request.form.get('farmer_issues', '').strip()

#         if farmer_issues:
#             query_type = predict_query_type(farmer_issues)
#             recommendations = recommendation(farmer_issues,query_type)
#             return render_template('result2.html', recommendations=recommendations, services=services,service_images=service_images)
        
#         else:
#             return render_template('index.html', error="Please enter farmer issues.")
        
#     return render_template('index.html')




@app.route('/api/services')
def get_services():
    # return jsonify([{"name": name, "description": desc} for name, desc in services.items()])
    return jsonify([
        {
            "name": name,
            "description": desc,
            "image": service_images.get(name, "/static/images/default.jpg")  # Default image if not found
        } 
        for name, desc in services.items()
    ])

@app.route('/services')
def services_page():
    return render_template('services.html')

@app.route('/services/<service_name>')
def service_page(service_name):
    formatted_service = service_name.replace("_", " ")  # Convert URL format to match dictionary keys

    if formatted_service == "Rental Equipment Facility":
        return redirect("http://127.0.0.1:5001/")
    # if formatted_service == "Insurance Service":
    #     return render_template(f"services/insurance_services/insurance.html")
    if formatted_service in services:
        return render_template(f"services/{service_name}.html")
    else:
        return "<h2>Service not found</h2>", 404

@app.route('/<form_name>_form')
def render_form(form_name):
    form_template = f"services/{form_name}_form.html"
    try:
        return render_template(form_template)
    except:
        return "Form not found", 404

if __name__ == '__main__':
    app.run(debug=True)
