import os
import json
import boto3
import urllib.request
import urllib.error
from datetime import datetime
import uuid
# from mcp.server.mcpserver import MCPServer
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Customer Retention Copilot Server")
# mcp = MCPServer("Customer Retention Copilot Server")

dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name='eu-north-1')

CUSTOMERS_TABLE = os.environ.get("CUSTOMERS_TABLE", "Customers_data")
CASES_TABLE = os.environ.get("CASES_TABLE", "Cases")

LAMBDA_URL = "https://ht3aj24syezugyou5rzu3dav4a0rpzzg.lambda-url.eu-north-1.on.aws/predict"

KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "DYR4BM5HTV")
MODEL_ARN = os.environ.get("MODEL_ARN", "eu.amazon.nova-pro-v1:0")

CUSTOM_PROMPT = """
You are a precise telecom customer policy assistant. Answer using ONLY the provided search results.

- Identify the exact requirement in the user's question.
- Find ALL roles, levels, or categories whose stated limits satisfy that requirement.
- Evaluate EVERY candidate in the retrieved context before answering.
- Do not stop after finding the first matching candidate.
- Do not return only the highest-level candidate.
- Consider all applicable conditions, restrictions, and exceptions.
- Return every candidate satisfying this comparison.
- A candidate is eligible when the required value is less than or equal to their stated maximum limit.
- Evaluate each candidate independently. Do not exclude a candidate merely because another candidate has a higher maximum limit.
- Do not confuse eligibility criteria with approval authority roles. Return every candidate satisfying this comparison.

- Never invent information. If the context is insufficient, say "I am not aware of it".

Search Results:
$search_results$

User Question:
$query$

Answer:
"""


@mcp.tool()
def get_customer(customer_id: str) -> str:
    """Retrieves customer profile information including tenure, services, and charges using customer ID.
    Always call this first when a user asks about a specific customer."""

    table = dynamodb.Table(CUSTOMERS_TABLE)
    response = table.get_item(Key={'customerID': customer_id})
    item = response.get('Item')
    
    if not item:
        return json.dumps({"error": f"Customer {customer_id} not found."})
    return json.dumps(item, default=str, indent=2)


@mcp.tool()
def predict_churn(customer_id: str) -> str:
    """Predicts customer churn probability, risk percentage, and risk category (Low, Medium, High).
    Call this after retrieving customer info to evaluate their churn risk."""

    table = dynamodb.Table(CUSTOMERS_TABLE)
    response = table.get_item(Key={'customerID': customer_id})
    item = response.get('Item')
    
    if not item:
        return json.dumps({"error": f"Customer {customer_id} not found."})
    
    model_payload = dict(item)
    model_payload.pop('Churn', None)
    # model_payload.pop('customerID', None)
    
    try:
        req = urllib.request.Request(
            LAMBDA_URL,
            data=json.dumps(model_payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        with urllib.request.urlopen(req) as resp:
            lambda_data = json.loads(resp.read().decode('utf-8'))
            
        if "predictions" in lambda_data and len(lambda_data["predictions"]) > 0:
            pred = lambda_data["predictions"][0]
        else:
            pred = lambda_data

        probability = float(pred.get("churn_probability"))

        # Calculate business risk category
        if probability < 0.30:
            risk_category = "Low"
        elif probability < 0.70:
            risk_category = "Medium"
        else:
            risk_category = "High"

        return json.dumps({
            "customer_id": customer_id,
            "churn_probability": probability,
            "churn_risk_percentage": round(probability * 100, 2),
            "predicted_churn": pred.get("predicted_churn"),
            "risk_category": risk_category
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to reach Lambda prediction endpoint: {str(e)}"})

@mcp.tool()
def search_policies(query: str) -> str:
    """Searches the retention policy knowledge base for rules, offers, policies, eligibility criteria, and guidelines.
    
    Use this tool in two scenarios:
    1. Standalone queries: When the user asks general questions about policy documents.
    2. Multi-step workflows: After retrieving a customer's details and churn risk score, use this tool to search for specific retention offers matching their risk level.
    """

    try:
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={
                'text': query
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                    'modelArn': MODEL_ARN,
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 3  # Custom top 3 chunks (default is 5)
                        }
                    },
                    'generationConfiguration': {
                        'promptTemplate': {
                            'textPromptTemplate': CUSTOM_PROMPT
                        }
                    }
                }
            }
        )
        
        answer_text = response.get('output', {}).get('text', 'No answer generated.')
        citations = response.get('citations', [])
        
        return json.dumps({
            "query": query,
            "policy_answer": answer_text,
            "citations_count": len(citations)
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to execute retrieve_and_generate: {str(e)}"})


@mcp.tool()
def create_case(customer_id: str, issue_description: str, risk_category: str) -> str:
    """Creates a formal retention case record in DynamoDB for a customer. 
    Only call this tool when the user explicitly requests to create, open, or register a case."""

   
    cust_table = dynamodb.Table(CUSTOMERS_TABLE)
    cust_response = cust_table.get_item(Key={'customerID': customer_id})
    if not cust_response.get('Item'):
        return json.dumps({"error": f"Cannot process case: Customer {customer_id} does not exist."})

    cases_table = dynamodb.Table(CASES_TABLE)
    
    # 2. Scan the cases table to see if an active case already exists for this customer
    existing_case_response = cases_table.scan(
        FilterExpression=Attr('customer_id').eq(customer_id)
    )
    
    items = existing_case_response.get('Items', [])
    # Sort by created_at descending to find the most recent case
    items.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    # Filter for open/active cases (modify statuses as per your business logic)
    open_cases = [c for c in items if c.get('status') not in ['Closed', 'Resolved']]

    if open_cases:
        # Update the existing open case instead of creating a new row
        case_item = open_cases[0]
        case_id = case_item['case_id']
        
        if "High" in risk_category:
            case_status = "Mandatory Manager Approval"
        elif "Medium" in risk_category:
            case_status = "Standard Agent Review"
        else:
            case_status = case_item.get('status', 'Auto-approved')

        cases_table.update_item(
            Key={'case_id': case_id},
            UpdateExpression="SET issue_description = :desc, risk_category = :risk, #st = :status, updated_at = :upd",
            ExpressionAttributeNames={'#st': 'status'},
            ExpressionAttributeValues={
                ':desc': issue_description,
                ':risk': risk_category,
                ':status': case_status,
                ':upd': datetime.utcnow().isoformat()
            }
        )
        action = "updated"
        message = f"Existing open retention case {case_id} updated successfully with status: {case_status}"
    else:
        # Create a brand new case if no open cases exist
        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        
        if "High" in risk_category:
            case_status = "Mandatory Manager Approval"
        elif "Medium" in risk_category:
            case_status = "Standard Agent Review"
        else:
            case_status = "Auto-approved"

        case_item = {
            "case_id": case_id,
            "customer_id": customer_id,
            "issue_description": issue_description,
            "risk_category": risk_category,
            "status": case_status,
            "created_at": datetime.utcnow().isoformat()
        }
        cases_table.put_item(Item=case_item)
        action = "created"
        message = f"New retention case registered successfully with status: {case_status}"

    return json.dumps({
        "success": True,
        "action": action,
        "case_id": case_id,
        "customer_id": customer_id,
        "risk_category": risk_category,
        "case_status": case_status,
        "message": message
    }, indent=2)




if __name__ == "__main__":
    mcp.run()