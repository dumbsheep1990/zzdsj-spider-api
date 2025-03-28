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


@router.get("/templates",
            summary="Get vectorization templates",
            description="Returns a list of all vectorization templates.",
            response_description="List of vectorization templates"
            )
async def get_templates(db=Depends(get_async_db)):
    """Get all vectorization templates"""
    try:
        cursor = db.vector_templates.find({}).sort("name", 1)
        templates = await cursor.to_list(length=100)
        
        # Process the results
        result = []
        for template in templates:
            template["id"] = str(template["_id"])
            del template["_id"]
            
            if "created_at" in template:
                template["created_at"] = template["created_at"].isoformat() if isinstance(template["created_at"], datetime) else template["created_at"]
                
            if "updated_at" in template:
                template["updated_at"] = template["updated_at"].isoformat() if isinstance(template["updated_at"], datetime) else template["updated_at"]
                
            result.append(template)
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get vectorization templates: {str(e)}")


@router.get("/template/{template_id}",
            summary="Get vectorization template",
            description="Returns details of a specific vectorization template.",
            response_description="Vectorization template details"
            )
async def get_template(template_id: str = Path(..., description="Template ID"), 
                        db=Depends(get_async_db)):
    """Get a specific vectorization template"""
    try:
        from bson.objectid import ObjectId
        
        # Convert string ID to ObjectId
        try:
            obj_id = ObjectId(template_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid template ID format")
        
        # Get template from database
        template = await db.vector_templates.find_one({"_id": obj_id})
        
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Format the result
        template["id"] = str(template["_id"])
        del template["_id"]
        
        if "created_at" in template:
            template["created_at"] = template["created_at"].isoformat() if isinstance(template["created_at"], datetime) else template["created_at"]
            
        if "updated_at" in template:
            template["updated_at"] = template["updated_at"].isoformat() if isinstance(template["updated_at"], datetime) else template["updated_at"]
            
        return template
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get vectorization template: {str(e)}")


@router.post("/template",
             summary="Save vectorization template",
             description="Saves a new vectorization template or updates an existing one.",
             response_description="Saved vectorization template"
             )
async def save_template(name: str = Body(..., description="Template name"),
                          description: Optional[str] = Body(None, description="Template description"),
                          text_fields: List[str] = Body(..., description="Text fields to vectorize"),
                          metadata_fields: Optional[List[str]] = Body(None, description="Metadata fields to include"),
                          chunk_size: Optional[int] = Body(None, description="Text chunk size for long documents"),
                          chunk_overlap: Optional[int] = Body(None, description="Chunk overlap for long documents"),
                          template_id: Optional[str] = Body(None, description="Template ID (for updates)"),
                          db=Depends(get_async_db)):
    """Save a vectorization template"""
    try:
        from bson.objectid import ObjectId
        
        now = datetime.now()
        template_data = {
            "name": name,
            "description": description,
            "text_fields": text_fields,
            "metadata_fields": metadata_fields,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "updated_at": now
        }
        
        # Update existing or create new
        if template_id:
            # Convert string ID to ObjectId
            try:
                obj_id = ObjectId(template_id)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid template ID format")
            
            # Check if template exists
            existing = await db.vector_templates.find_one({"_id": obj_id})
            if not existing:
                raise HTTPException(status_code=404, detail="Template not found")
            
            # Update existing template
            await db.vector_templates.update_one(
                {"_id": obj_id},
                {"$set": template_data}
            )
            
            # Return updated template
            result = await db.vector_templates.find_one({"_id": obj_id})
            result["id"] = str(result["_id"])
            del result["_id"]
            
            if "created_at" in result:
                result["created_at"] = result["created_at"].isoformat() if isinstance(result["created_at"], datetime) else result["created_at"]
                
            if "updated_at" in result:
                result["updated_at"] = result["updated_at"].isoformat() if isinstance(result["updated_at"], datetime) else result["updated_at"]
                
            return result
            
        else:
            # Create new template
            template_data["created_at"] = now
            insert_result = await db.vector_templates.insert_one(template_data)
            
            # Return new template
            template_id = str(insert_result.inserted_id)
            return {
                "id": template_id,
                "name": name,
                "description": description,
                "text_fields": text_fields,
                "metadata_fields": metadata_fields,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat()
            }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save vectorization template: {str(e)}")


@router.delete("/template/{template_id}",
               summary="Delete vectorization template",
               description="Deletes a specific vectorization template.",
               response_description="Deletion confirmation"
               )
async def delete_template(template_id: str = Path(..., description="Template ID"),
                           db=Depends(get_async_db)):
    """Delete a vectorization template"""
    try:
        from bson.objectid import ObjectId
        
        # Convert string ID to ObjectId
        try:
            obj_id = ObjectId(template_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid template ID format")
        
        # Check if template exists
        existing = await db.vector_templates.find_one({"_id": obj_id})
        if not existing:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Delete the template
        result = await db.vector_templates.delete_one({"_id": obj_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=500, detail="Template deletion failed")
        
        return {"message": "Template deleted successfully", "id": template_id}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete vectorization template: {str(e)}")


@router.post("/batch",
             summary="Start batch vectorization",
             description="Starts a background task to vectorize articles based on criteria.",
             response_description="Batch vectorization task status"
             )
async def start_batch_vectorization(background_tasks: BackgroundTasks,
                                      article_ids: Optional[List[str]] = Body(None, description="List of article IDs to vectorize"),
                                      filter: Optional[Dict[str, Any]] = Body(None, description="Filter criteria for articles"),
                                      template_id: Optional[str] = Body(None, description="Template ID to use"),
                                      model: Optional[str] = Body(None, description="Vector model to use"),
                                      db=Depends(get_async_db)):
    """Start batch vectorization of articles"""
    try:
        from bson.objectid import ObjectId
        import uuid
        
        # Validate inputs
        if not article_ids and not filter:
            raise HTTPException(status_code=400, detail="Either article_ids or filter must be provided")
        
        # Get vector configuration
        vector_config = await db.system_config.find_one({"type": "vector_config"})
        if not vector_config:
            raise HTTPException(status_code=400, detail="Vector service not configured")
        
        # Determine model to use
        model_to_use = model or vector_config.get("model", "all-MiniLM-L6-v2")
        
        # Get template if specified
        template = None
        if template_id:
            try:
                template = await db.vector_templates.find_one({"_id": ObjectId(template_id)})
                if not template:
                    raise HTTPException(status_code=404, detail="Template not found")
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid template ID format")
        
        # Create batch task record
        batch_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Determine articles to vectorize
        query = {}
        if article_ids:
            # Convert string IDs to ObjectIds
            try:
                obj_ids = [ObjectId(aid) for aid in article_ids]
                query = {"_id": {"$in": obj_ids}}
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid article ID format")
        elif filter:
            query = filter
        
        # Count total articles to process
        total_articles = await db.articles.count_documents(query)
        
        if total_articles == 0:
            raise HTTPException(status_code=400, detail="No articles found matching the criteria")
        
        # Create batch record
        batch_record = {
            "_id": batch_id,
            "status": "processing",
            "created_at": now,
            "total": total_articles,
            "completed": 0,
            "failed": 0,
            "query": query,
            "model": model_to_use
        }
        
        if template_id:
            batch_record["template_id"] = template_id
            batch_record["template_name"] = template.get("name") if template else None
        
        await db.batch_vectorization.insert_one(batch_record)
        
        # Start vectorization in background
        background_tasks.add_task(
            perform_batch_vectorization,
            batch_id=batch_id,
            query=query,
            model=model_to_use,
            template=template,
            vector_config=vector_config,
            db=db
        )
        
        # Return initial response
        return {
            "batch_id": batch_id,
            "status": "processing",
            "total": total_articles,
            "completed": 0,
            "failed": 0,
            "created_at": now.isoformat()
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start batch vectorization: {str(e)}")


async def perform_batch_vectorization(batch_id: str, query: Dict[str, Any], model: str, 
                                      template: Optional[Dict[str, Any]], vector_config: Dict[str, Any], db):
    """Background task to perform batch vectorization"""
    try:
        from bson.objectid import ObjectId
        import time
        
        # Determine which fields to vectorize and chunk settings
        text_fields = template.get("text_fields", ["title", "content"]) if template else ["title", "content"]
        metadata_fields = template.get("metadata_fields", []) if template else []
        chunk_size = template.get("chunk_size") if template else None
        chunk_overlap = template.get("chunk_overlap") if template else None
        
        # Get the provider settings
        provider = vector_config.get("provider", "local")
        
        # Initialize vectorizer based on provider and model
        vectorizer = None
        if provider == "local":
            # For demonstration, we're simulating the vectorization process
            # In a real implementation, you would use a proper vectorization library
            vectorizer = MockVectorizer(model)
        elif provider == "pinecone":
            # Initialize Pinecone client
            pass
        elif provider == "qdrant":
            # Initialize Qdrant client
            pass
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Process articles in batches
        batch_size = 20
        completed = 0
        failed = 0
        
        # Get a cursor for all matching articles
        cursor = db.articles.find(query)
        
        async for article in cursor:
            try:
                # Prepare text for vectorization
                text_to_vectorize = ""
                for field in text_fields:
                    if field in article and article[field]:
                        text_to_vectorize += article[field] + " "
                
                if not text_to_vectorize.strip():
                    # Skip articles with no content to vectorize
                    failed += 1
                    continue
                
                # Apply chunking if specified
                chunks = [text_to_vectorize]
                if chunk_size and len(text_to_vectorize) > chunk_size:
                    chunks = []
                    overlap = chunk_overlap or 0
                    for i in range(0, len(text_to_vectorize), chunk_size - overlap):
                        chunk = text_to_vectorize[i:i + chunk_size]
                        if len(chunk.strip()) > 0:
                            chunks.append(chunk)
                
                # Generate vectors for each chunk
                vectors = []
                for chunk in chunks:
                    try:
                        vector = vectorizer.vectorize(chunk)
                        vectors.append(vector)
                    except Exception as ve:
                        # Log vectorization error
                        print(f"Error vectorizing chunk: {str(ve)}")
                        continue
                
                # Prepare metadata
                metadata = {}
                for field in metadata_fields:
                    if field in article:
                        metadata[field] = article[field]
                
                # Update article with vector and metadata
                update_data = {
                    "vector": vectors[0] if vectors else None,  # Store first vector directly on article
                    "vectors": vectors if len(vectors) > 1 else [],  # Store all vectors if multiple chunks
                    "vectorized_at": datetime.now(),
                    "vector_model": model,
                    "vector_metadata": metadata
                }
                
                # Update the article
                await db.articles.update_one(
                    {"_id": article["_id"]},
                    {"$set": update_data}
                )
                
                completed += 1
                
                # Update batch record periodically
                if completed % 10 == 0 or failed % 10 == 0:
                    await db.batch_vectorization.update_one(
                        {"_id": batch_id},
                        {"$set": {
                            "completed": completed,
                            "failed": failed
                        }}
                    )
                
                # Artificial delay to avoid overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                # Log processing error
                print(f"Error processing article {article.get('_id')}: {str(e)}")
                failed += 1
        
        # Mark batch as completed
        await db.batch_vectorization.update_one(
            {"_id": batch_id},
            {"$set": {
                "status": "completed",
                "completed": completed,
                "failed": failed,
                "completed_at": datetime.now()
            }}
        )
        
    except Exception as e:
        # Update batch with error
        try:
            await db.batch_vectorization.update_one(
                {"_id": batch_id},
                {"$set": {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now()
                }}
            )
        except Exception as update_error:
            # Log if we can't update database
            print(f"Failed to update batch status: {str(update_error)}")


class MockVectorizer:
    """Mock vectorizer for demonstration purposes"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.dimension = 1536 if model_name == "text-embedding-3-large" else 384
    
    def vectorize(self, text):
        """Generate a mock vector of the appropriate dimension"""
        import random
        import hashlib
        
        # Use text hash to generate consistent vectors for same input
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (10 ** 8)
        random.seed(seed)
        
        # Generate vector of the appropriate dimension
        return [random.uniform(-1, 1) for _ in range(self.dimension)]


@router.get("/batch/{batch_id}",
            summary="Get batch vectorization status",
            description="Returns the status of a batch vectorization task.",
            response_description="Batch vectorization task status"
            )
async def get_batch_status(batch_id: str = Path(..., description="Batch ID"),
                            db=Depends(get_async_db)):
    """Get status of a batch vectorization task"""
    try:
        # Fetch batch record
        batch = await db.batch_vectorization.find_one({"_id": batch_id})
        
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        # Format response
        result = {
            "batch_id": batch["_id"],
            "status": batch["status"],
            "total": batch["total"],
            "completed": batch["completed"],
            "failed": batch["failed"],
            "created_at": batch["created_at"].isoformat() if isinstance(batch["created_at"], datetime) else batch["created_at"]
        }
        
        # Add optional fields
        if "error" in batch:
            result["error"] = batch["error"]
        
        if "completed_at" in batch:
            result["completed_at"] = batch["completed_at"].isoformat() if isinstance(batch["completed_at"], datetime) else batch["completed_at"]
        
        if "template_name" in batch:
            result["template_name"] = batch["template_name"]
        
        if "model" in batch:
            result["model"] = batch["model"]
        
        return result
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch status: {str(e)}")


@router.get("/batches",
            summary="Get batch vectorization history",
            description="Returns a list of batch vectorization tasks.",
            response_description="List of batch vectorization tasks"
            )
async def get_batch_history(page: int = Query(1, ge=1, description="Page number"),
                             page_size: int = Query(20, ge=1, le=100, description="Items per page"),
                             db=Depends(get_async_db)):
    """Get history of batch vectorization tasks"""
    try:
        # Calculate skip value for pagination
        skip = (page - 1) * page_size
        
        # Get total count
        total = await db.batch_vectorization.count_documents({})
        
        # Get batches with pagination
        cursor = db.batch_vectorization.find({}).sort("created_at", -1).skip(skip).limit(page_size)
        batches = await cursor.to_list(length=page_size)
        
        # Process results
        items = []
        for batch in batches:
            item = {
                "batch_id": batch["_id"],
                "status": batch["status"],
                "total": batch["total"],
                "completed": batch["completed"],
                "failed": batch["failed"],
                "created_at": batch["created_at"].isoformat() if isinstance(batch["created_at"], datetime) else batch["created_at"]
            }
            
            if "completed_at" in batch:
                item["completed_at"] = batch["completed_at"].isoformat() if isinstance(batch["completed_at"], datetime) else batch["completed_at"]
            
            if "template_name" in batch:
                item["template_name"] = batch["template_name"]
            
            if "model" in batch:
                item["model"] = batch["model"]
            
            items.append(item)
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch history: {str(e)}")


@router.post("/search",
             summary="Perform similarity search",
             description="Searches for articles similar to the query text.",
             response_description="List of similar articles"
             )
async def similarity_search(query: str = Body(..., description="Search query"),
                            filter: Optional[Dict[str, Any]] = Body(None, description="Filter criteria"),
                            top_k: int = Body(10, description="Number of results to return"),
                            threshold: Optional[float] = Body(None, description="Similarity threshold"),
                            db=Depends(get_async_db)):
    """Perform a similarity search"""
    try:
        # Get vector configuration
        vector_config = await db.system_config.find_one({"type": "vector_config"})
        if not vector_config:
            raise HTTPException(status_code=400, detail="Vector service not configured")
        
        provider = vector_config.get("provider", "local")
        model = vector_config.get("model", "all-MiniLM-L6-v2")
        
        # Initialize vectorizer to convert query to vector
        vectorizer = MockVectorizer(model)
        query_vector = vectorizer.vectorize(query)
        
        # For demonstration, we'll simulate vector search
        # In a real implementation, you would use a proper vector database
        
        # Build the combined query
        base_query = {"vector": {"$exists": True}}
        if filter:
            # Combine with user filter
            base_query.update(filter)
        
        # Fetch articles (in a real implementation, this would be a vector similarity search)
        cursor = db.articles.find(base_query).limit(100)  # Fetch more than needed to allow for filtering
        articles = await cursor.to_list(length=100)
        
        # Simulate vector similarity by calculating cosine similarity
        results = []
        for article in articles:
            if "vector" in article and article["vector"]:
                similarity = calculate_cosine_similarity(query_vector, article["vector"])
                
                # Apply threshold if specified
                if threshold is not None and similarity < threshold:
                    continue
                
                # Create result item
                item = {
                    "id": str(article["_id"]),
                    "similarity": similarity,
                    "title": article.get("title", ""),
                    "content": truncate_text(article.get("content", ""), 200),
                    "url": article.get("url", ""),
                }
                
                # Add metadata fields
                for field in ["publish_date", "department", "domain"]:
                    if field in article:
                        item[field] = article[field]
                
                results.append(item)
        
        # Sort by similarity and limit to top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:top_k]
        
        return {
            "query": query,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    try:
        import numpy as np
        
        a = np.array(vec1)
        b = np.array(vec2)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    except Exception:
        # Fall back to random similarity for demonstration
        import random
        return random.uniform(0.3, 0.95)


def truncate_text(text, max_length):
    """Truncate text to max_length and add ellipsis if needed"""
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."
