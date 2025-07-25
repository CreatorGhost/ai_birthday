import os
import time
from typing import List, TypedDict, Annotated
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.tools import tool

from .config import ModelConfig

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    from pinecone import Pinecone
    from pinecone.core.client.models import ServerlessSpec

load_dotenv()

class RAGState(TypedDict):
    """
    Represents the state of our RAG graph.
    
    Attributes:
        question: User's question
        chat_history: Previous conversation messages
        documents: Retrieved documents
        generation: LLM generation/answer
        search_needed: Whether web search is needed
        loop_step: Current iteration step
        source_documents: Source documents for the answer
    """
    question: str
    chat_history: List[dict]
    documents: List[Document]
    generation: str
    search_needed: str
    loop_step: int
    source_documents: List[Document]

class RAGPipeline:
    """Advanced RAG pipeline using LangGraph for complex AI agent workflows"""
    
    def __init__(self):
        # Initialize model configuration
        self.model_config = ModelConfig()
        
        # Get API keys and configuration
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'faq-embeddings')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize embeddings and LLM using model configuration
        self.embeddings = self.model_config.create_embedding_model()
        self.llm = self.model_config.create_chat_llm()
        
        # LLM for document grading (using same model with temperature 0)
        chat_config = self.model_config.get_chat_model_config().copy()
        chat_config['temperature'] = 0
        
        if self.model_config.llm_provider.value == 'openai':
            from langchain_openai import ChatOpenAI
            self.llm_json_mode = ChatOpenAI(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                model=chat_config["model_name"],
                temperature=chat_config["temperature"],
                max_tokens=chat_config.get("max_tokens")
            )
        else:  # Google Gemini
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm_json_mode = ChatGoogleGenerativeAI(
                    google_api_key=os.getenv('GOOGLE_API_KEY'),
                    model=chat_config["model_name"],
                    temperature=chat_config["temperature"],
                    max_output_tokens=chat_config.get("max_output_tokens")
                )
            except ImportError:
                raise ImportError(
                    "langchain-google-genai is required for Google Gemini models. "
                    "Install it with: pip install langchain-google-genai"
                )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.vector_store = None
        self.retriever = None
        self.graph = None
        
        # Setup prompts
        self._setup_prompts()
        
        # Print configuration info
        provider_info = self.model_config.get_provider_info()
        print(f"Initialized RAG Pipeline with:")
        print(f"  Provider: {provider_info['provider']}")
        print(f"  Chat Model: {provider_info['chat_model']}")
        print(f"  Embedding Model: {provider_info['embedding_model']}")
        
    def _setup_prompts(self):
        """Setup all prompts used in the RAG pipeline"""
        
        # RAG generation prompt with chat history support
        self.rag_prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks.
            
            Use the following documents to answer the question.
            
            If you don't know the answer, just say that you don't know.
            
            Use three sentences maximum and keep the answer concise.
            
            Consider the chat history for context, but focus on answering the current question:
            
            Chat History: {chat_history}
            
            Context: {context}
            Question: {question}
            Answer: """,
            input_variables=["context", "question", "chat_history"]
        )
        
        # Document grader instructions
        self.doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
        
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        
        Respond with only 'yes' or 'no' to indicate whether the document is relevant."""
        
        # Document grader prompt
        self.doc_grader_prompt = """Here is the retrieved document: 
        
        {document} 
        
        Here is the user question: 
        
        {question}
        
        Carefully assess whether the document contains information relevant to the question.
        
        Respond with only 'yes' or 'no'."""
    
    def setup_pinecone_index(self, dimension=None):
        """Create or connect to Pinecone index"""
        try:
            # Use model-specific embedding dimensions if not provided
            if dimension is None:
                dimension = self.model_config.get_embedding_dimensions()
            
            # Check if index exists
            if self.index_name not in [index.name for index in self.pc.list_indexes()]:
                print(f"Creating new Pinecone index: {self.index_name} with dimension {dimension}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.pinecone_environment
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
            else:
                print(f"Using existing Pinecone index: {self.index_name}")
            
            return True
        except Exception as e:
            print(f"Error setting up Pinecone index: {str(e)}")
            return False
    
    def create_vector_store(self, documents):
        """Create vector store from documents using Pinecone"""
        try:
            self.vector_store = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name
            )
            
            # Setup retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            print(f"Created vector store with {len(documents)} documents")
            return True
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return False
    
    def load_existing_vector_store(self):
        """Load existing vector store from Pinecone"""
        try:
            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings
            )
            
            # Setup retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            print("Loaded existing vector store")
            return True
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def _create_retriever_tool(self):
        """Create a retriever tool for the graph"""
        @tool
        def retriever_tool(query: str) -> List[Document]:
            """Retrieve documents from the vector store based on the query."""
            if not self.retriever:
                raise ValueError("Retriever not initialized")
            return self.retriever.invoke(query)
        
        return retriever_tool
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for context"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _format_chat_history(self, chat_history: List[dict]) -> str:
        """Format chat history for context"""
        if not chat_history:
            return "No previous conversation."
        
        formatted_history = []
        for msg in chat_history[-5:]:  # Keep last 5 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_history.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history)
    
    # Graph Nodes
    def retrieve_documents(self, state: RAGState) -> RAGState:
        """
        Retrieve documents from vectorstore
        
        Args:
            state: The current graph state
            
        Returns:
            Updated state with retrieved documents
        """
        print("---RETRIEVE DOCUMENTS---")
        question = state["question"]
        
        # Retrieve documents
        documents = self.retriever.invoke(question)
        
        return {
            "documents": documents,
            "question": question,
            "chat_history": state.get("chat_history", []),
            "loop_step": state.get("loop_step", 0)
        }
    
    def grade_documents(self, state: RAGState) -> RAGState:
        """
        Determines whether the retrieved documents are relevant to the question
        
        Args:
            state: The current graph state
            
        Returns:
            Updated state with filtered documents and search decision
        """
        print("---GRADE DOCUMENTS---")
        question = state["question"]
        documents = state["documents"]
        
        # Score each document
        filtered_docs = []
        search_needed = "No"
        
        for doc in documents:
            doc_grader_prompt_formatted = self.doc_grader_prompt.format(
                document=doc.page_content, 
                question=question
            )
            
            result = self.llm_json_mode.invoke([
                SystemMessage(content=self.doc_grader_instructions),
                HumanMessage(content=doc_grader_prompt_formatted)
            ])
            
            grade = result.content.strip().lower()
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            elif grade == "no":
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                search_needed = "Yes"
            else:
                print(f"---GRADE: UNEXPECTED RESPONSE '{grade}', KEEPING DOCUMENT---")
                filtered_docs.append(doc)
        
        return {
            "documents": filtered_docs,
            "question": question,
            "chat_history": state.get("chat_history", []),
            "search_needed": search_needed,
            "loop_step": state.get("loop_step", 0)
        }
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using RAG on retrieved documents
        
        Args:
            state: The current graph state
            
        Returns:
            Updated state with generated answer
        """
        print("---GENERATE ANSWER---")
        question = state["question"]
        documents = state["documents"]
        chat_history = state.get("chat_history", [])
        loop_step = state.get("loop_step", 0)
        
        # Format documents for context
        context = self._format_docs(documents)
        
        # Format chat history for context
        chat_history_str = self._format_chat_history(chat_history)
        
        # Generate answer using RAG chain
        rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        generation = rag_chain.invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history_str
        })
        
        return {
            "documents": documents,
            "question": question,
            "chat_history": chat_history,
            "generation": generation,
            "source_documents": documents,
            "loop_step": loop_step + 1
        }
    
    def decide_to_generate(self, state: RAGState) -> str:
        """
        Determines whether to generate an answer or need more search
        
        Args:
            state: The current graph state
            
        Returns:
            Next node to call
        """
        search_needed = state.get("search_needed", "No")
        
        if search_needed == "Yes":
            print("---DECISION: DOCUMENTS NOT RELEVANT, NEED BETTER RETRIEVAL---")
            # For now, we'll generate anyway since we don't have web search
            # In a full implementation, you could add web search here
            return "generate"
        else:
            print("---DECISION: GENERATE ANSWER---")
            return "generate"
    
    def setup_graph(self):
        """Setup the LangGraph workflow"""
        if not self.retriever:
            raise ValueError("Vector store and retriever must be initialized first")
        
        # Create the graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate_answer)
        
        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge based on document relevance
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "generate": "generate"
            }
        )
        
        workflow.add_edge("generate", END)
        
        # Compile the graph
        self.graph = workflow.compile()
        
        print("LangGraph RAG workflow setup complete")
    
    def answer_question(self, question: str, chat_history: List[dict] = None) -> dict:
        """
        Answer a question using the LangGraph RAG pipeline
        
        Args:
            question: User's question
            chat_history: Previous conversation messages for context
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.graph:
            raise ValueError("Graph not initialized. Call setup_graph() first.")
        
        if chat_history is None:
            chat_history = []
        
        try:
            # Initial state
            initial_state = {
                "question": question,
                "chat_history": chat_history,
                "documents": [],
                "generation": "",
                "search_needed": "No",
                "loop_step": 0,
                "source_documents": []
            }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            return {
                "answer": result.get("generation", "No answer generated"),
                "source_documents": result.get("source_documents", [])
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "source_documents": []
            }
    
    def get_graph_visualization(self):
        """Get a visualization of the graph structure"""
        if not self.graph:
            raise ValueError("Graph not initialized")
        
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            return f"Error generating graph visualization: {str(e)}"