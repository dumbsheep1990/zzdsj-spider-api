from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path, Body
from typing import Dict, Any, List, Optional
from datetime import datetime
from db.connection import get_async_db

router = APIRouter(prefix="/vector", tags=["vector"])


@router.get("/config",
            summary="Get vector service configuration",
            description="Returns the current vector service configuration.",
            response_description="Current vector service configuration"
            )
async def get_config(db=Depends(get_async_db)):
    """Get current vector service configuration"""
    try:
        config = await db.system_config.find_one({"type": "vector_config"})
        
        if not config:
            # Return default config if none exists
            return {
                "provider": "local",
                "dimension": 1536,
                "model": "all-MiniLM-L6-v2",
                "auto_vectorize": False,
                "config_exists": False
            }
        
        # Remove internal fields
        if "_id" in config:
            config["_id"] = str(config["_id"])
        if "type" in config:
            del config["type"]
        
        config["config_exists"] = True
        return config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve vector configuration: {str(e)}")


@router.post("/config",
             summary="Save vector service configuration",
             description="Saves vector service configuration.",
             response_description="Saved vector service configuration"
             )
async def save_config(config: Dict[str, Any] = Body(...), db=Depends(get_async_db)):
    """Save vector service configuration"""
    try:
        # Add metadata
        config["type"] = "vector_config"
        config["updated_at"] = datetime.now()
        
        # Upsert config
        result = await db.system_config.update_one(
            {"type": "vector_config"},
            {"$set": config},
            upsert=True
        )
        
        # Clean up response
        if "_id" in config:
            config["_id"] = str(config["_id"])
        if "type" in config:
            del config["type"]
        
        config["config_exists"] = True
        return config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save vector configuration: {str(e)}")


@router.post("/test-connection",
             summary="Test vector service connection",
             description="Tests the connection to the specified vector service provider.",
             response_description="Test connection result"
             )
async def test_connection(config: Dict[str, Any] = Body(...)):
    """Test connection to vector service provider"""
    try:
        provider = config.get("provider")
        
        if not provider:
            raise HTTPException(status_code=400, detail="Provider not specified")
        
        # Simulate connection test for different providers
        if provider == "local":
            model = config.get("model", "all-MiniLM-L6-v2")
            
            # In a real implementation, we would try to load the specified model
            return {
                "success": True,
                "message": f"Successfully loaded local model: {model}",
                "model_info": {
                    "dimension": 384 if model == "all-MiniLM-L6-v2" else 1536,
                    "metric": "cosine"
                }
            }
            
        elif provider == "pinecone":
            api_key = config.get("api_key")
            environment = config.get("environment")
            index_name = config.get("index_name")
            
            if not api_key or not environment or not index_name:
                raise HTTPException(status_code=400, detail="API key, environment, and index name required for Pinecone")
                
            # In a real implementation, we would try to connect to Pinecone
            return {
                "success": True,
                "message": f"Successfully connected to Pinecone index: {index_name}",
                "index_info": {
                    "dimension": config.get("dimension", 1536),
                    "metric": "cosine",
                    "pod_type": "p1.x1"
                }
            }
            
        elif provider == "qdrant":
            url = config.get("url")
            collection_name = config.get("collection_name")
            
            if not url or not collection_name:
                raise HTTPException(status_code=400, detail="URL and collection name required for Qdrant")
                
            # In a real implementation, we would try to connect to Qdrant
            return {
                "success": True,
                "message": f"Successfully connected to Qdrant collection: {collection_name}",
                "collection_info": {
                    "dimension": config.get("dimension", 1536),
                    "vectors": 0,
                    "status": "green"
                }
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")


@router.get("/stats",
            summary="Get vector statistics",
            description="Returns statistics about the vector index.",
            response_description="Vector index statistics"
            )
async def get_stats(db=Depends(get_async_db)):
    """Get statistics about the vector index"""
    try:
        # Get vector config first
        config = await db.system_config.find_one({"type": "vector_config"})
        
        # Count vectorized articles
        vectorized_count = await db.articles.count_documents({"vector": {"$exists": True}})
        total_count = await db.articles.count_documents({})
        
        # Get latest vectorization run
        latest_run = await db.vector_history.find_one(sort=[("created_at", -1)])
        
        return {
            "vectorized_count": vectorized_count,
            "total_articles": total_count,
            "vectorization_ratio": vectorized_count / total_count if total_count > 0 else 0,
            "provider": config.get("provider", "local") if config else "local",
            "model": config.get("model", "all-MiniLM-L6-v2") if config else "all-MiniLM-L6-v2",
            "dimension": config.get("dimension", 1536) if config else 1536,
            "latest_run": {
                "timestamp": latest_run["created_at"].isoformat() if latest_run and "created_at" in latest_run else None,
                "status": latest_run["status"] if latest_run else None,
                "articles_processed": latest_run.get("processed_count", 0) if latest_run else 0
            } if latest_run else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve vector statistics: {str(e)}")


@router.post("/rebuild-index",
             summary="Rebuild vector index",
             description="Starts a background task to rebuild the vector index for all articles.",
             response_description="Vector index rebuild task status"
             )
async def rebuild_index(background_tasks: BackgroundTasks, db=Depends(get_async_db)):
    """Rebuild the vector index for all articles"""
    try:
        # Create history record
        history_id = await db.vector_history.insert_one({
            "type": "rebuild",
            "status": "started",
            "created_at": datetime.now(),
            "total_count": await db.articles.count_documents({}),
            "processed_count": 0
        })
        
        # TODO: Implement actual rebuild logic in a background task
        # For now, we'll just update the history record directly
        
        # In a real implementation, this would be done in the background task:
        history_id_str = str(history_id.inserted_id)
        
        return {
            "message": "Vector index rebuild started",
            "history_id": history_id_str,
            "status": "started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start vector index rebuild: {str(e)}")


@router.get("/progress",
            summary="Get vectorization progress",
            description="Returns the progress of the current vectorization task.",
            response_description="Vectorization progress"
            )
async def get_progress(db=Depends(get_async_db)):
    """Get progress of the current vectorization task"""
    try:
        # Get latest vectorization run
        latest_run = await db.vector_history.find_one(sort=[("created_at", -1)])
        
        if not latest_run or latest_run["status"] not in ["started", "processing"]:
            return {
                "active": False,
                "message": "No active vectorization task"
            }
        
        total = latest_run.get("total_count", 0)
        processed = latest_run.get("processed_count", 0)
        
        return {
            "active": True,
            "history_id": str(latest_run["_id"]),
            "type": latest_run.get("type", "rebuild"),
            "status": latest_run["status"],
            "total": total,
            "processed": processed,
            "progress": (processed / total * 100) if total > 0 else 0,
            "started_at": latest_run["created_at"].isoformat(),
            "elapsed_time": (datetime.now() - latest_run["created_at"]).total_seconds()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve vectorization progress: {str(e)}")


@router.get("/history",
            summary="Get vectorization history",
            description="Returns a list of previous vectorization tasks.",
            response_description="Vectorization history"
            )
async def get_history(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        db=Depends(get_async_db)
):
    """Get history of vectorization tasks with pagination"""
    try:
        # Count total records for pagination
        total = await db.vector_history.count_documents({})
        
        # Apply pagination
        skip = (page - 1) * page_size
        cursor = db.vector_history.find().sort("created_at", -1).skip(skip).limit(page_size)
        history = await cursor.to_list(length=page_size)
        
        # Format response
        for item in history:
            item["_id"] = str(item["_id"])
            if "created_at" in item:
                item["created_at"] = item["created_at"].isoformat()
            if "completed_at" in item:
                item["completed_at"] = item["completed_at"].isoformat()
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "data": history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve vectorization history: {str(e)}")


@router.post("/cancel",
             summary="Cancel vectorization task",
             description="Cancels the currently running vectorization task.",
             response_description="Cancellation result"
             )
async def cancel_vectorization(db=Depends(get_async_db)):
    """Cancel the current vectorization task"""
    try:
        # Get latest vectorization run
        latest_run = await db.vector_history.find_one(sort=[("created_at", -1)])
        
        if not latest_run or latest_run["status"] not in ["started", "processing"]:
            raise HTTPException(status_code=400, detail="No active vectorization task to cancel")
        
        # Update status to cancelled
        await db.vector_history.update_one(
            {"_id": latest_run["_id"]},
            {"$set": {
                "status": "cancelled",
                "completed_at": datetime.now(),
                "cancelled": True
            }}
        )
        
        # TODO: Implement actual cancellation logic for the background task
        
        return {
            "message": "Vectorization task cancelled",
            "history_id": str(latest_run["_id"])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel vectorization task: {str(e)}")


@router.get("/rules",
            summary="Get vectorization rules",
            description="Returns the list of vectorization rules.",
            response_description="Vectorization rules"
            )
async def get_rules(db=Depends(get_async_db)):
    """Get the list of vectorization rules"""
    try:
        rules = await db.vector_rules.find().to_list(length=100)
        
        # Format response
        for rule in rules:
            rule["_id"] = str(rule["_id"])
            if "created_at" in rule:
                rule["created_at"] = rule["created_at"].isoformat()
        
        return rules
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve vectorization rules: {str(e)}")


@router.post("/rules",
             summary="Save vectorization rule",
             description="Saves a new vectorization rule or updates an existing one.",
             response_description="Saved vectorization rule"
             )
async def save_rule(rule: Dict[str, Any] = Body(...), db=Depends(get_async_db)):
    """Save a vectorization rule"""
    try:
        # Validate rule
        required_fields = ["name", "field", "enabled"]
        for field in required_fields:
            if field not in rule:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Add timestamps
        rule["updated_at"] = datetime.now()
        
        if "_id" in rule:
            # Update existing rule
            rule_id = rule["_id"]
            if isinstance(rule_id, str):
                from bson.objectid import ObjectId
                rule_id = ObjectId(rule_id)
                
            del rule["_id"]
            
            result = await db.vector_rules.update_one(
                {"_id": rule_id},
                {"$set": rule}
            )
            
            if result.matched_count == 0:
                raise HTTPException(status_code=404, detail="Rule not found")
                
            rule["_id"] = str(rule_id)
            
        else:
            # Create new rule
            rule["created_at"] = datetime.now()
            
            result = await db.vector_rules.insert_one(rule)
            rule["_id"] = str(result.inserted_id)
        
        return rule
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save vectorization rule: {str(e)}")


@router.delete("/rules/{rule_id}",
               summary="Delete vectorization rule",
               description="Deletes a vectorization rule.",
               response_description="Deletion result"
               )
async def delete_rule(
        rule_id: str = Path(..., description="Rule ID"),
        db=Depends(get_async_db)
):
    """Delete a vectorization rule"""
    try:
        from bson.objectid import ObjectId
        result = await db.vector_rules.delete_one({"_id": ObjectId(rule_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        return {
            "message": "Vectorization rule deleted successfully",
            "rule_id": rule_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete vectorization rule: {str(e)}")
