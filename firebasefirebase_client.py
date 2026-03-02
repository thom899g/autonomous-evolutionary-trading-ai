"""
Firebase Firestore client for persistent storage of strategies, metrics, and state
Handles all database operations with robust error handling
"""
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import Client as FirestoreClient
from google.cloud.firestore_v1.base_query import FieldFilter

from config import settings

logger = logging.getLogger(__name__)

class FirebaseClient:
    """Firebase Firestore client for trading system"""
    
    _instance = None
    _db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize Firebase connection"""
        try:
            if not firebase_admin._apps:
                if settings.FIREBASE_CREDENTIALS_PATH:
                    cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
                else:
                    # Try environment variable
                    cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
                    if cred_json:
                        cred_dict = json.loads(cred_json)
                        cred = credentials.Certificate(cred_dict)
                    else:
                        raise ValueError("No Firebase credentials provided")
                
                firebase_admin.initialize_app(cred, {
                    'projectId': settings.FIREBASE_PROJECT_ID
                })
            
            self._db = firestore.client()
            logger.info("Firebase Firestore initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise
    
    @property
    def db(self) -> FirestoreClient:
        """Get Firestore database instance"""
        if self._db is None:
            self._initialize()
        return self._db
    
    def save_strategy(self, strategy_id: str, strategy_data: Dict[str, Any]) -> bool:
        """Save or update a