"""
Chatbox helper module for RNA structure analysis using RCSB PDB data and Azure OpenAI.
"""

import requests
import json
from typing import Dict, List, Optional, Any
import os


def fetch_rcsb_data(pdb_id: str) -> Dict[str, Any]:
    """
    Fetch comprehensive PDB information from RCSB PDB API.
    
    Args:
        pdb_id: PDB identifier (e.g., "2GDI")
        
    Returns:
        Dictionary containing structure information
    """
    data = {
        'pdb_id': pdb_id,
        'title': None,
        'description': None,
        'experimental_method': None,
        'resolution': None,
        'authors': [],
        'citation': None,
        'doi': None,
        'pubmed_id': None,
        'pubmed_abstract': None,
        'ligands': [],
        'biological_function': None
    }
    
    try:
        # Fetch main entry data
        api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            entry_data = response.json()
            
            # Basic structure information
            data['title'] = entry_data.get('struct', {}).get('title', 'Title not available')
            data['description'] = entry_data.get('struct', {}).get('pdbx_descriptor', '')
            
            # Experimental method and resolution
            exptl = entry_data.get('exptl', [])
            if exptl and len(exptl) > 0:
                data['experimental_method'] = exptl[0].get('method', 'Unknown')
            
            refine = entry_data.get('refine', [])
            if refine and len(refine) > 0:
                resolution = refine[0].get('ls_d_res_high')
                if resolution:
                    data['resolution'] = f"{resolution} Å"
            
            # PubMed ID for fetching abstract
            citations = entry_data.get('rcsb_primary_citation', {})
            if citations:
                data['pubmed_id'] = citations.get('pdbx_database_id_PubMed')
                data['doi'] = citations.get('pdbx_database_id_DOI')
                
                # Citation info
                journal = citations.get('journal_abbrev', '')
                year = citations.get('year')
                title = citations.get('title', '')
                data['citation'] = f"{title} {journal} ({year})" if title and year else None
        
        # Fetch ligand information
        try:
            ligand_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}"
            ligand_response = requests.get(ligand_url, timeout=10)
            if ligand_response.status_code == 200:
                ligand_data = ligand_response.json()
                if isinstance(ligand_data, list):
                    for ligand in ligand_data:
                        chem_comp = ligand.get('rcsb_nonpolymer_entity_container_identifiers', {})
                        ligand_id = chem_comp.get('comp_id')
                        ligand_name = ligand.get('pdbx_description', '')
                        if ligand_id and ligand_id not in ['HOH', 'WAT']:  # Skip water
                            data['ligands'].append({
                                'id': ligand_id,
                                'name': ligand_name
                            })
        except Exception as e:
            print(f"Error fetching ligand data: {e}")
        
        # Fetch PubMed abstract if available
        if data['pubmed_id']:
            try:
                data['pubmed_abstract'] = fetch_pubmed_abstract(data['pubmed_id'])
            except Exception as e:
                print(f"Error fetching PubMed abstract: {e}")
        
        # Fetch authors from PubMed citation
        try:
            pub_url = f"https://data.rcsb.org/rest/v1/core/pubmed/{pdb_id}"
            pub_response = requests.get(pub_url, timeout=10)
            if pub_response.status_code == 200:
                pub_data = pub_response.json()
                authors = pub_data.get('rcsb_pubmed_container_identifiers', {}).get('pubmed_id')
                if authors:
                    # This endpoint may have author info
                    pass
        except Exception as e:
            print(f"Error fetching publication data: {e}")
            
    except Exception as e:
        print(f"Error fetching RCSB data for {pdb_id}: {e}")
    
    return data


def fetch_pubmed_abstract(pubmed_id: str) -> Optional[str]:
    """
    Fetch abstract from PubMed using NCBI E-utilities API.
    
    Args:
        pubmed_id: PubMed identifier
        
    Returns:
        Abstract text or None if not available
    """
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': pubmed_id,
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            # Parse XML response to extract abstract
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # Find abstract text
            abstract_elem = root.find('.//Abstract/AbstractText')
            if abstract_elem is not None and abstract_elem.text:
                return abstract_elem.text
            
            # Try alternative path
            abstract_texts = root.findall('.//Abstract/AbstractText')
            if abstract_texts:
                # Combine all abstract sections
                return ' '.join([elem.text for elem in abstract_texts if elem.text])
        
        return None
    except Exception as e:
        print(f"Error fetching PubMed abstract for {pubmed_id}: {e}")
        return None


def create_system_prompt(rcsb_data: Dict[str, Any]) -> str:
    """
    Create system prompt for Azure OpenAI with RCSB PDB context.
    
    Args:
        rcsb_data: Dictionary containing RCSB PDB information
        
    Returns:
        System prompt string
    """
    prompt = """You are an expert RNA structural biologist assistant. You have access to detailed information about an RNA structure from the RCSB Protein Data Bank.

Your role is to:
1. Provide clear, accurate information about this RNA structure
2. Answer questions about structural features, experimental methods, biological function, and ligand interactions
3. Use the provided PDB data as your primary source, supplemented with your general knowledge of RNA biology
4. Be conversational but scientifically accurate
5. When you don't have specific information in the provided data, acknowledge this and provide general context

Here is the information about the current RNA structure:
"""
    
    # Add PDB ID and title
    prompt += f"\n**PDB ID:** {rcsb_data['pdb_id']}"
    if rcsb_data['title']:
        prompt += f"\n**Title:** {rcsb_data['title']}"
    
    # Add experimental details
    if rcsb_data['experimental_method']:
        prompt += f"\n**Experimental Method:** {rcsb_data['experimental_method']}"
    if rcsb_data['resolution']:
        prompt += f"\n**Resolution:** {rcsb_data['resolution']}"
    
    # Add citation
    if rcsb_data['citation']:
        prompt += f"\n**Citation:** {rcsb_data['citation']}"
    if rcsb_data['doi']:
        prompt += f"\n**DOI:** {rcsb_data['doi']}"
    
    # Add abstract if available
    if rcsb_data['pubmed_abstract']:
        prompt += f"\n\n**Research Abstract:**\n{rcsb_data['pubmed_abstract']}"
    
    # Add ligand information
    if rcsb_data['ligands']:
        prompt += "\n\n**Bound Ligands:**"
        for ligand in rcsb_data['ligands']:
            prompt += f"\n- {ligand['id']}: {ligand['name']}"
    
    # Add description
    if rcsb_data['description']:
        prompt += f"\n\n**Description:** {rcsb_data['description']}"
    
    prompt += "\n\nRemember to be helpful, accurate, and cite the specific data when relevant."
    
    return prompt


def generate_introduction_prompt(rcsb_data: Dict[str, Any]) -> str:
    """
    Generate a prompt to create an introduction message about the RNA structure.
    
    Args:
        rcsb_data: Dictionary containing RCSB PDB information
        
    Returns:
        User prompt requesting introduction
    """
    prompt = f"""Please provide a brief, friendly introduction to this RNA structure (PDB ID: {rcsb_data['pdb_id']}). 

Keep it concise (3-4 sentences) and highlight:
- What this structure represents
- The experimental method used
- Key findings or significance

Then, end with: "I can answer questions about:" followed by a bullet list of 4-5 specific topics users might want to explore, such as:
- Experimental details and structure quality
- Ligand binding and interactions
- Biological function and mechanism
- Structural features
- Research applications and significance"""
    
    return prompt


def call_azure_openai(
    messages: List[Dict[str, str]], 
    api_key: str, 
    endpoint: str,
    deployment: str = "gpt-4"
) -> Optional[str]:
    """
    Call Azure OpenAI Chat Completions API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        api_key: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint URL
        deployment: Deployment name (default: "gpt-4")
        
    Returns:
        Assistant's response text or None if error
    """
    try:
        from openai import AzureOpenAI
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-12-01-preview",
            azure_endpoint=endpoint
        )
        
        # Make the API call
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,  # ~150-200 words
            top_p=1.0
        )
        
        # Extract the assistant's response
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        
        return None
        
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        raise e


def generate_introduction(rcsb_data: Dict[str, Any], api_key: str, endpoint: str, deployment: str = "gpt-4") -> str:
    """
    Generate introduction message using Azure OpenAI.
    
    Args:
        rcsb_data: Dictionary containing RCSB PDB information
        api_key: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint URL
        deployment: Deployment name (default: "gpt-4")
        
    Returns:
        Introduction message text
    """
    try:
        system_prompt = create_system_prompt(rcsb_data)
        user_prompt = generate_introduction_prompt(rcsb_data)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = call_azure_openai(messages, api_key, endpoint, deployment)
        
        if response:
            return response
        else:
            # Fallback message if API fails
            return generate_fallback_introduction(rcsb_data)
            
    except Exception as e:
        print(f"Error generating introduction: {e}")
        return generate_fallback_introduction(rcsb_data)


def generate_fallback_introduction(rcsb_data: Dict[str, Any]) -> str:
    """
    Generate a basic introduction without AI when API is unavailable.
    
    Args:
        rcsb_data: Dictionary containing RCSB PDB information
        
    Returns:
        Basic introduction text
    """
    intro = f"Welcome! This is the structure of {rcsb_data['title']} (PDB ID: {rcsb_data['pdb_id']})."
    
    if rcsb_data['experimental_method']:
        intro += f" The structure was determined using {rcsb_data['experimental_method']}"
        if rcsb_data['resolution']:
            intro += f" at {rcsb_data['resolution']} resolution"
        intro += "."
    
    if rcsb_data['ligands']:
        ligand_names = [lig['id'] for lig in rcsb_data['ligands'][:3]]
        intro += f" This structure includes bound ligands: {', '.join(ligand_names)}."
    
    intro += "\n\nI can answer questions about:\n"
    intro += "- Experimental details and structure quality\n"
    intro += "- Ligand binding and interactions\n"
    intro += "- Biological function and mechanism\n"
    intro += "- Structural features\n"
    intro += "- Research applications and significance"
    
    return intro

