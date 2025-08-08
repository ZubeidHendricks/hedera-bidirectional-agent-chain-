#!/usr/bin/env python3
"""
Hedera RAG Knowledge Base Creation Script
Creates embeddings, FAISS database, and knowledge base for Hedera RAG agent
Uses Gemini Embedding model with 3072 dimensions and normalization
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Please install: pip install google-genai")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("⚠️ FAISS not installed. Creating embeddings without FAISS index.")
    faiss = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hedera_embeddings_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HederaKnowledgeBaseCreator:
    """
    Creates and manages Hedera knowledge base with embeddings and FAISS index
    """
    
    def __init__(self, kb_directory: str = "agents/hedera_rag_kb"):
        self.kb_directory = Path(kb_directory)
        self.kb_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize Gemini client
        self._initialize_genai_client()
        
        # Configuration
        self.embedding_dimensions = 3072
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # File paths
        self.embeddings_file = self.kb_directory / "hedera_embeddings.pkl"
        self.faiss_index_file = self.kb_directory / "hedera_faiss_index.bin"
        self.metadata_file = self.kb_directory / "hedera_metadata.json"
        self.knowledge_base_file = self.kb_directory / "hedera_knowledge_base.json"
        
        logger.info(f"🔧 Hedera Knowledge Base Creator initialized")
        logger.info(f"📁 Knowledge base directory: {self.kb_directory}")
        logger.info(f"🔢 Embedding dimensions: {self.embedding_dimensions}")
    
    def _initialize_genai_client(self):
        """Initialize Google GenAI client"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            self.client = genai.Client(api_key=api_key)
            logger.info("✅ Gemini client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            raise
    
    def load_hedera_data(self) -> Dict[str, pd.DataFrame]:
        """Load all Hedera CSV files and return as dictionary"""
        data_files = {}
        
        csv_files = list(self.kb_directory.glob("*.csv"))
        logger.info(f"📊 Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                file_key = csv_file.stem.replace("Hedera Dataset - ", "").replace(" ", "_").lower()
                data_files[file_key] = df
                logger.info(f"✅ Loaded {csv_file.name} ({len(df)} rows)")
            except Exception as e:
                logger.error(f"❌ Failed to load {csv_file.name}: {e}")
        
        return data_files
    
    def create_text_chunks(self, data_files: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Create text chunks from Hedera data for embedding - now with automatic column detection"""
        chunks = []
        
        for file_key, df in data_files.items():
            logger.info(f"📝 Processing {file_key} with columns: {list(df.columns)}")
            
            # Auto-detect file type based on columns and filename patterns
            file_type = self._detect_file_type(file_key, df.columns.tolist())
            logger.info(f"🔍 Detected file type: {file_type}")
            
            # Create chunks based on detected type
            if file_type == "developer_guides":
                chunks.extend(self._create_developer_guide_chunks_auto(df, file_key))
            elif file_type == "services":
                chunks.extend(self._create_services_chunks_auto(df, file_key))
            elif file_type == "api_examples":
                chunks.extend(self._create_api_examples_chunks_auto(df, file_key))
            elif file_type == "use_cases":
                chunks.extend(self._create_use_cases_chunks_auto(df, file_key))
            elif file_type == "documentation":
                chunks.extend(self._create_documentation_chunks_auto(df, file_key))
            elif file_type == "tutorials":
                chunks.extend(self._create_tutorials_chunks_auto(df, file_key))
            else:
                chunks.extend(self._create_generic_chunks_auto(df, file_key))
        
        logger.info(f"📄 Created {len(chunks)} text chunks")
        return chunks
    
    def _detect_file_type(self, file_key: str, columns: List[str]) -> str:
        """Automatically detect file type based on filename and columns"""
        file_key_lower = file_key.lower()
        columns_lower = [col.lower() for col in columns]
        
        # Check filename patterns first
        if "developer" in file_key_lower or "guides" in file_key_lower:
            return "developer_guides"
        elif "services" in file_key_lower or "hcs" in file_key_lower or "hfs" in file_key_lower:
            return "services"
        elif "api" in file_key_lower or "examples" in file_key_lower:
            return "api_examples"
        elif "use_cases" in file_key_lower or "usecases" in file_key_lower:
            return "use_cases"
        elif "documentation" in file_key_lower or "docs" in file_key_lower:
            return "documentation"
        elif "tutorials" in file_key_lower or "tutorial" in file_key_lower:
            return "tutorials"
        
        # Check column patterns if filename doesn't match
        if any(col in columns_lower for col in ["developer", "guide", "sdk", "integration"]):
            return "developer_guides"
        elif any(col in columns_lower for col in ["service", "hcs", "hfs", "hts", "consensus"]):
            return "services"
        elif any(col in columns_lower for col in ["api", "endpoint", "method", "response"]):
            return "api_examples"
        elif any(col in columns_lower for col in ["use_case", "application", "industry"]):
            return "use_cases"
        elif any(col in columns_lower for col in ["documentation", "description", "concept"]):
            return "documentation"
        elif any(col in columns_lower for col in ["tutorial", "step", "walkthrough"]):
            return "tutorials"
        
        return "generic"
    
    def _get_column_value(self, row: pd.Series, possible_names: List[str], default: str = "Unknown") -> str:
        """Get value from row using possible column names"""
        for name in possible_names:
            if name in row.index and pd.notna(row[name]):
                return str(row[name])
        return default
    
    def _create_developer_guide_chunks_auto(self, df: pd.DataFrame, file_key: str) -> List[Dict[str, Any]]:
        """Create chunks from developer guides data with automatic column detection"""
        chunks = []
        
        for _, row in df.iterrows():
            # Flexible column mapping for Hedera developer guides
            guide_title = self._get_column_value(row, ["title", "guide_title", "name"])
            description = self._get_column_value(row, ["description", "summary", "overview"])
            category = self._get_column_value(row, ["category", "type", "section"])
            difficulty = self._get_column_value(row, ["difficulty", "level", "complexity"])
            sdk_version = self._get_column_value(row, ["sdk_version", "version", "hedera_version"])
            
            # Create comprehensive developer guide text
            guide_text = f"HEDERA DEVELOPER GUIDE: {guide_title}"
            guide_text += f"\nCategory: {category}"
            guide_text += f"\nDifficulty: {difficulty}"
            
            if sdk_version != "Unknown":
                guide_text += f"\nSDK Version: {sdk_version}"
            
            guide_text += f"\nDescription: {description}"
            
            # Add additional guide information if available
            additional_fields = {
                "prerequisites": ["prerequisites", "requirements", "prereq"],
                "estimated_time": ["estimated_time", "duration", "time"],
                "technologies": ["technologies", "tech_stack", "tools"],
                "code_language": ["code_language", "language", "programming_language"],
                "hedera_services": ["hedera_services", "services", "hcs_hfs_hts"],
                "use_case": ["use_case", "application", "scenario"]
            }
            
            for field_name, possible_columns in additional_fields.items():
                value = self._get_column_value(row, possible_columns)
                if value != "Unknown":
                    field_display = field_name.replace("_", " ").title()
                    guide_text += f"\n{field_display}: {value}"
            
            chunks.append({
                "text": guide_text,
                "metadata": {
                    "file_key": file_key,
                    "chunk_type": "developer_guide",
                    "guide_title": guide_title,
                    "category": category,
                    "difficulty": difficulty,
                    "sdk_version": sdk_version
                }
            })
        
        return chunks
    
    def _create_services_chunks_auto(self, df: pd.DataFrame, file_key: str) -> List[Dict[str, Any]]:
        """Create chunks from Hedera services data with automatic column detection"""
        chunks = []
        
        # Create overall standings summary
        standings_text = f"MOTOGP 2025 CHAMPIONSHIP STANDINGS:\nTotal Riders: {len(df)}"
        
        if len(df) > 0:
            # Get leader info
            leader = df.iloc[0]
            leader_name = self._build_rider_name(leader)
            leader_number = self._get_column_value(leader, ["racing_number", "number"])
            leader_points = self._get_column_value(leader, ["points"])
            leader_team = self._get_column_value(leader, ["team", "racing_team"])
            
            standings_text += f"\n\nCurrent Leader: {leader_name}"
            if leader_number != "Unknown":
                standings_text += f" (#{leader_number})"
            standings_text += f"\nLeader Points: {leader_points}\nLeader Team: {leader_team}"
            
            standings_text += "\n\nTOP 10 RIDERS:"
            
            for i, row in df.head(10).iterrows():
                rider_name = self._build_rider_name(row)
                position = self._get_column_value(row, ["position", "pos", "rank"])
                number = self._get_column_value(row, ["racing_number", "number"])
                points = self._get_column_value(row, ["points"])
                team = self._get_column_value(row, ["team", "racing_team"])
                points_gap = self._get_column_value(row, ["points_gap", "gap"])
                
                standings_text += f"\n{position}. {rider_name}"
                if number != "Unknown":
                    standings_text += f" (#{number})"
                standings_text += f" - {points} points"
                standings_text += f"\n   Team: {team}"
                if points_gap != "Unknown":
                    standings_text += f"\n   Points Gap: {points_gap}"
        
        chunks.append({
            "text": standings_text.strip(),
            "metadata": {
                "file_key": file_key,
                "chunk_type": "standings_summary",
                "season": "2025"
            }
        })
        
        # Create individual rider standings
        for _, row in df.iterrows():
            rider_name = self._build_rider_name(row)
            position = self._get_column_value(row, ["position", "pos", "rank"])
            number = self._get_column_value(row, ["racing_number", "number"])
            points = self._get_column_value(row, ["points"])
            team = self._get_column_value(row, ["team", "racing_team"])
            points_gap = self._get_column_value(row, ["points_gap", "gap"])
            
            rider_standing = f"RIDER STANDING: {rider_name}"
            if number != "Unknown":
                rider_standing += f" (#{number})"
            rider_standing += f"\nChampionship Position: {position}\nPoints: {points}"
            if points_gap != "Unknown":
                rider_standing += f"\nPoints Gap to Leader: {points_gap}"
            rider_standing += f"\nTeam: {team}"
            
            chunks.append({
                "text": rider_standing.strip(),
                "metadata": {
                    "file_key": file_key,
                    "chunk_type": "rider_standing",
                    "rider_name": rider_name,
                    "racing_number": number,
                    "position": position,
                    "points": points,
                    "team": team
                }
            })
        
        return chunks
    
    def _create_api_examples_chunks_auto(self, df: pd.DataFrame, file_key: str) -> List[Dict[str, Any]]:
        """Create chunks from API examples data with automatic column detection"""
        chunks = []
        
        # Extract race name from file key
        race_name = file_key.replace("results_", "").replace("2025_", "").replace("_", " ").title()
        
        # Create race summary
        if len(df) > 0:
            winner = df.iloc[0]
            winner_name = self._build_rider_name(winner)
            winner_number = self._get_column_value(winner, ["racing_number", "number"])
            winner_team = self._get_column_value(winner, ["team", "racing_team"])
            winner_time = self._get_column_value(winner, ["time_gap", "time", "total_time"])
            winner_points = self._get_column_value(winner, ["points"])
            
            race_summary = f"RACE RESULTS: {race_name} 2025\nWinner: {winner_name}"
            if winner_number != "Unknown":
                race_summary += f" (#{winner_number})"
            race_summary += f"\nWinner Team: {winner_team}"
            if winner_time != "Unknown":
                race_summary += f"\nWinner Time: {winner_time}"
            race_summary += f"\nTotal Finishers: {len(df)}"
            if winner_points != "Unknown":
                race_summary += f"\nPoints Awarded: {winner_points}"
            
            race_summary += "\n\nPODIUM FINISHERS:"
            
            # Add podium info
            for i, row in df.head(3).iterrows():
                rider_name = self._build_rider_name(row)
                position = self._get_column_value(row, ["position", "pos"])
                number = self._get_column_value(row, ["racing_number", "number"])
                team = self._get_column_value(row, ["team", "racing_team"])
                time_gap = self._get_column_value(row, ["time_gap", "gap"])
                points = self._get_column_value(row, ["points"])
                
                race_summary += f"\n{position}. {rider_name}"
                if number != "Unknown":
                    race_summary += f" (#{number})"
                race_summary += f" - {team}"
                if time_gap != "Unknown":
                    race_summary += f"\n   Time Gap: {time_gap}"
                if points != "Unknown":
                    race_summary += f"\n   Points: {points}"
            
            chunks.append({
                "text": race_summary.strip(),
                "metadata": {
                    "file_key": file_key,
                    "chunk_type": "race_summary",
                    "race_name": race_name,
                    "season": "2025",
                    "winner": winner_name,
                    "winner_team": winner_team
                }
            })
        
        # Create individual result entries
        for _, row in df.iterrows():
            rider_name = self._build_rider_name(row)
            position = self._get_column_value(row, ["position", "pos"])
            number = self._get_column_value(row, ["racing_number", "number"])
            team = self._get_column_value(row, ["team", "racing_team"])
            time_gap = self._get_column_value(row, ["time_gap", "gap", "time"])
            points = self._get_column_value(row, ["points"])
            
            result_text = f"RACE RESULT: {race_name} 2025\nPosition: {position}\nRider: {rider_name}"
            if number != "Unknown":
                result_text += f" (#{number})"
            result_text += f"\nTeam: {team}"
            if time_gap != "Unknown":
                result_text += f"\nTime Gap: {time_gap}"
            if points != "Unknown":
                result_text += f"\nPoints Scored: {points}"
            
            chunks.append({
                "text": result_text.strip(),
                "metadata": {
                    "file_key": file_key,
                    "chunk_type": "race_result",
                    "race_name": race_name,
                    "season": "2025",
                    "rider_name": rider_name,
                    "racing_number": number,
                    "position": position,
                    "points": points,
                    "team": team
                }
            })
        
        return chunks
    
    def _create_use_cases_chunks_auto(self, df: pd.DataFrame, file_key: str) -> List[Dict[str, Any]]:
        """Create chunks from use cases data with automatic column detection"""
        chunks = []
        
        # Create season calendar summary
        calendar_summary = f"MOTOGP 2025 SEASON CALENDAR:\nTotal Races: {len(df)}"
        
        if len(df) > 0:
            first_race = df.iloc[0]
            last_race = df.iloc[-1]
            
            first_date = self._get_column_value(first_race, ["date"])
            first_event = self._get_column_value(first_race, ["event", "race", "grand_prix"])
            last_date = self._get_column_value(last_race, ["date"])
            last_event = self._get_column_value(last_race, ["event", "race", "grand_prix"])
            
            calendar_summary += f"\nSeason Start: {first_date} - {first_event}"
            calendar_summary += f"\nSeason End: {last_date} - {last_event}"
            calendar_summary += "\n\nCOMPLETE RACE SCHEDULE:"
            
            for i, row in df.iterrows():
                event = self._get_column_value(row, ["event", "race", "grand_prix"])
                date = self._get_column_value(row, ["date"])
                track = self._get_column_value(row, ["track", "circuit", "venue"])
                
                calendar_summary += f"\n{i+1}. {event}\n   Date: {date}\n   Track: {track}"
        
        chunks.append({
            "text": calendar_summary.strip(),
            "metadata": {
                "file_key": file_key,
                "chunk_type": "season_calendar",
                "season": "2025",
                "total_races": len(df)
            }
        })
        
        # Create individual race event chunks
        for i, row in df.iterrows():
            event = self._get_column_value(row, ["event", "race", "grand_prix"])
            date = self._get_column_value(row, ["date"])
            track = self._get_column_value(row, ["track", "circuit", "venue"])
            
            race_event = f"RACE EVENT: {event}\nRound: {i+1}\nDate: {date}\nCircuit: {track}"
            
            chunks.append({
                "text": race_event.strip(),
                "metadata": {
                    "file_key": file_key,
                    "chunk_type": "race_event",
                    "event_name": event,
                    "date": date,
                    "track": track,
                    "round": i+1
                }
            })
        
        return chunks
    
    def _create_documentation_chunks_auto(self, df: pd.DataFrame, file_key: str) -> List[Dict[str, Any]]:
        """Create chunks from documentation data with automatic column detection"""
        chunks = []
        
        # Create records summary
        records_summary = f"""MOTOGP HISTORICAL RECORDS AND STATISTICS:
Total Records: {len(df)}

CATEGORIES COVERED:
- Speed Records
- Lap Records  
- Championship Records
- Pole Position Records
- Victory Records
- Career Milestones"""
        
        chunks.append({
            "text": records_summary.strip(),
            "metadata": {
                "file_key": file_key,
                "chunk_type": "records_summary",
                "total_records": len(df)
            }
        })
        
        # Create individual record entries
        for _, row in df.iterrows():
            event = self._get_column_value(row, ["event", "record_type", "category"])
            name = self._get_column_value(row, ["name", "holder", "rider"])
            speed = self._get_column_value(row, ["speed_kilometers_per_hour", "speed", "kmh"])
            time = self._get_column_value(row, ["time", "lap_time"])
            age = self._get_column_value(row, ["age"])
            count = self._get_column_value(row, ["count", "number", "quantity"])
            date = self._get_column_value(row, ["date", "year"])
            track = self._get_column_value(row, ["track", "circuit", "venue"])
            
            record_text = f"RECORD: {event}\nRecord Holder: {name}"
            if speed != "Unknown":
                record_text += f"\nSpeed: {speed} km/h"
            if time != "Unknown":
                record_text += f"\nTime: {time}"
            if age != "Unknown":
                record_text += f"\nAge: {age}"
            if count != "Unknown":
                record_text += f"\nCount: {count}"
            record_text += f"\nDate: {date}\nCircuit: {track}"
            
            chunks.append({
                "text": record_text.strip(),
                "metadata": {
                    "file_key": file_key,
                    "chunk_type": "record_entry",
                    "record_event": event,
                    "record_holder": name,
                    "speed": speed,
                    "time": time,
                    "date": date,
                    "track": track
                }
            })
        
        return chunks
    
    def _create_tutorials_chunks_auto(self, df: pd.DataFrame, file_key: str) -> List[Dict[str, Any]]:
        """Create chunks from tutorials data with automatic column detection"""
        chunks = []
        
        for _, row in df.iterrows():
            event = self._get_column_value(row, ["event", "race", "grand_prix"])
            date = self._get_column_value(row, ["date"])
            track = self._get_column_value(row, ["track", "circuit", "venue"])
            weather = self._get_column_value(row, ["weather", "conditions"])
            air_temp = self._get_column_value(row, ["temperature_degrees_celcius", "temperature", "air_temp"])
            ground_temp = self._get_column_value(row, ["ground_temperature_degrees_celcius", "ground_temp", "track_temp"])
            track_condition = self._get_column_value(row, ["track_condition", "surface", "conditions"])
            humidity = self._get_column_value(row, ["humidity_percentage", "humidity"])
            
            weather_text = f"RACE CONDITIONS: {event}\nDate: {date}\nCircuit: {track}\nWeather: {weather}"
            if air_temp != "Unknown":
                weather_text += f"\nAir Temperature: {air_temp}°C"
            if ground_temp != "Unknown":
                weather_text += f"\nGround Temperature: {ground_temp}°C"
            weather_text += f"\nTrack Condition: {track_condition}"
            if humidity != "Unknown":
                weather_text += f"\nHumidity: {humidity}%"
            
            chunks.append({
                "text": weather_text.strip(),
                "metadata": {
                    "file_key": file_key,
                    "chunk_type": "race_conditions",
                    "event": event,
                    "date": date,
                    "track": track,
                    "weather": weather,
                    "temperature": air_temp,
                    "track_condition": track_condition,
                    "humidity": humidity
                }
            })
        
        return chunks
    
    def _create_generic_chunks_auto(self, df: pd.DataFrame, file_key: str) -> List[Dict[str, Any]]:
        """Create generic chunks for unknown file types with automatic column detection"""
        chunks = []
        
        # Create file summary
        summary_text = f"DATA FILE: {file_key}\nTotal Rows: {len(df)}\nColumns: {', '.join(df.columns.tolist())}"
        
        # Add sample data
        if len(df) > 0:
            summary_text += "\n\nSAMPLE DATA:"
            for i, row in df.head(3).iterrows():
                summary_text += f"\nRow {i+1}:"
                for col in df.columns[:5]:  # Limit to first 5 columns
                    value = str(row[col]) if pd.notna(row[col]) else "N/A"
                    summary_text += f"\n  {col}: {value}"
        
        chunks.append({
            "text": summary_text.strip(),
            "metadata": {
                "file_key": file_key,
                "chunk_type": "file_summary",
                "total_rows": len(df),
                "columns": df.columns.tolist()
            }
        })
        
        # Create individual row chunks for structured data
        for i, row in df.iterrows():
            if i >= 10:  # Limit to first 10 rows to avoid too many chunks
                break
                
            row_text = f"DATA ENTRY {i+1} from {file_key}:"
            for col in df.columns:
                value = str(row[col]) if pd.notna(row[col]) else "N/A"
                row_text += f"\n{col}: {value}"
            
            chunks.append({
                "text": row_text.strip(),
                "metadata": {
                    "file_key": file_key,
                    "chunk_type": "data_entry",
                    "row_index": i
                }
            })
        
        return chunks
    
    def _build_rider_name(self, row: pd.Series) -> str:
        """Build rider name from available columns"""
        first_name = self._get_column_value(row, ["first_name", "firstName"])
        last_name = self._get_column_value(row, ["last_name", "lastName", "surname"])
        
        if first_name != "Unknown" and last_name != "Unknown":
            return f"{first_name} {last_name}"
        elif first_name != "Unknown":
            return first_name
        elif last_name != "Unknown":
            return last_name
        else:
            return self._get_column_value(row, ["rider", "name", "rider_name"], "Unknown Rider")
    
    async def create_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Create embeddings for text chunks using Gemini"""
        embeddings = []
        processed_chunks = []
        
        logger.info(f"🔍 Creating embeddings for {len(chunks)} chunks...")
        
        # Process in batches for efficiency
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk["text"] for chunk in batch]
            
            try:
                # Create embeddings with Gemini
                result = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch_texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=self.embedding_dimensions
                    )
                )
                
                # Process embeddings
                for j, embedding in enumerate(result.embeddings):
                    embedding_array = np.array(embedding.values, dtype=np.float32)
                    
                    # Normalize the embedding (important for cosine similarity)
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm
                    
                    embeddings.append(embedding_array)
                    processed_chunks.append(batch[j])
                
                logger.info(f"✅ Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"❌ Error creating embeddings for batch {i//batch_size + 1}: {e}")
                # Continue with next batch
        
        logger.info(f"✅ Created {len(embeddings)} embeddings")
        return embeddings, processed_chunks
    
    def create_faiss_index(self, embeddings: List[np.ndarray]):
        """Create FAISS index for efficient similarity search"""
        if faiss is None:
            logger.warning("⚠️ FAISS not available, skipping index creation")
            return None
            
        logger.info("🔍 Creating FAISS index...")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index for normalized vectors
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        
        # Add vectors to index
        index.add(embeddings_array)
        
        logger.info(f"✅ FAISS index created with {index.ntotal} vectors")
        return index
    
    def save_knowledge_base(self, embeddings: List[np.ndarray], chunks: List[Dict[str, Any]], faiss_index):
        """Save all knowledge base components"""
        logger.info("💾 Saving knowledge base components...")
        
        # Save embeddings
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"✅ Saved embeddings to {self.embeddings_file}")
        
        # Save FAISS index if available
        if faiss_index is not None:
            faiss.write_index(faiss_index, str(self.faiss_index_file))
            logger.info(f"✅ Saved FAISS index to {self.faiss_index_file}")
        else:
            logger.info("⚠️ FAISS index not created, skipping save")
        
        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "embedding_dimensions": self.embedding_dimensions,
            "faiss_index_size": faiss_index.ntotal if faiss_index else 0,
            "chunk_types": list(set(chunk["metadata"]["chunk_type"] for chunk in chunks)),
            "files_processed": list(set(chunk["metadata"]["file_key"] for chunk in chunks))
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Saved metadata to {self.metadata_file}")
        
        # Save knowledge base (chunks without embeddings for reference)
        knowledge_base = {
            "chunks": chunks,
            "metadata": metadata
        }
        
        with open(self.knowledge_base_file, 'w') as f:
            json.dump(knowledge_base, f, indent=2, default=str)
        logger.info(f"✅ Saved knowledge base to {self.knowledge_base_file}")
    
    def test_knowledge_base(self, faiss_index, chunks: List[Dict[str, Any]]) -> None:
        """Test the knowledge base with sample queries"""
        logger.info("🧪 Testing knowledge base with sample queries...")
        
        test_queries = [
            "Who is leading the championship?",
            "Which rider has the most wins?",
            "What are the race results for Spain?",
            "Who is Marc Marquez?",
            "What is the race calendar for 2025?"
        ]
        
        for query in test_queries:
            try:
                # Create query embedding
                result = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=[query],
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=self.embedding_dimensions
                    )
                )
                
                query_embedding = np.array(result.embeddings[0].values, dtype=np.float32)
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                if faiss_index is not None:
                    # Search
                    k = 3
                    scores, indices = faiss_index.search(query_embedding.reshape(1, -1), k)
                    
                    logger.info(f"\n🔍 Query: {query}")
                    logger.info(f"Top {k} results:")
                    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                        if idx < len(chunks):
                            chunk = chunks[idx]
                            logger.info(f"  {i+1}. Score: {score:.4f} - {chunk['metadata']['chunk_type']}: {chunk['text'][:100]}...")
                else:
                    logger.info(f"\n🔍 Query: {query} - FAISS not available, skipping search test")
                
            except Exception as e:
                logger.error(f"❌ Error testing query '{query}': {e}")
    
    async def create_knowledge_base(self) -> None:
        """Main method to create the complete knowledge base"""
        logger.info("🚀 Starting Hedera Knowledge Base Creation...")
        
        try:
            # Step 1: Load data
            data_files = self.load_hedera_data()
            if not data_files:
                raise ValueError("No data files found")
            
            # Step 2: Create text chunks
            chunks = self.create_text_chunks(data_files)
            if not chunks:
                raise ValueError("No chunks created")
            
            # Step 3: Create embeddings
            embeddings, processed_chunks = await self.create_embeddings(chunks)
            if not embeddings:
                raise ValueError("No embeddings created")
            
            # Step 4: Create FAISS index
            faiss_index = self.create_faiss_index(embeddings)
            
            # Step 5: Save knowledge base
            self.save_knowledge_base(embeddings, processed_chunks, faiss_index)
            
            # Step 6: Test knowledge base
            self.test_knowledge_base(faiss_index, processed_chunks)
            
            logger.info("🎉 Hedera Knowledge Base Creation Completed Successfully!")
            logger.info(f"📊 Summary:")
            logger.info(f"   - Total chunks: {len(processed_chunks)}")
            logger.info(f"   - Embedding dimensions: {self.embedding_dimensions}")
            logger.info(f"   - FAISS index size: {faiss_index.ntotal if faiss_index else 0}")
            logger.info(f"   - Files processed: {len(data_files)}")
            
        except Exception as e:
            logger.error(f"❌ Knowledge base creation failed: {e}")
            raise

async def main():
    """Main function to run the knowledge base creation"""
    print("🌐 Hedera RAG Knowledge Base Creator")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("Please set your Google API key in the .env file")
        return
    
    # Create knowledge base
    creator = HederaKnowledgeBaseCreator()
    await creator.create_knowledge_base()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 