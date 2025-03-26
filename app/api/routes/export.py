from fastapi import APIRouter, HTTPException, Query, Depends, Response, Path
from typing import Optional, Dict, List, Any
from bson.objectid import ObjectId
from datetime import datetime
import csv
import io
import json
import pandas as pd
from db.connection import get_async_db

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/",
            summary="Export data",
            description="Exports data in various formats (json, csv, excel).",
            response_description="Exported data file"
            )
async def export_data(
        format: str = Query("json", description="Export format (json, csv, excel)"),
        collection: str = Query("articles", description="Data collection to export"),
        query: Optional[str] = Query(None, description="JSON query filter"),
        fields: Optional[str] = Query(None, description="Comma-separated list of fields to include"),
        limit: int = Query(1000, ge=1, le=5000, description="Maximum number of records to export"),
        db=Depends(get_async_db)
):
    """Export data in various formats"""
    if format not in ["json", "csv", "excel"]:
        raise HTTPException(status_code=400, detail="Unsupported export format")

    # Validate collection name
    valid_collections = ["articles", "urls", "ai_extractions"]
    if collection not in valid_collections:
        raise HTTPException(status_code=400, detail=f"Invalid collection. Must be one of: {', '.join(valid_collections)}")
    
    # Parse query if provided
    query_filter = {}
    if query:
        try:
            query_filter = json.loads(query)
        except:
            raise HTTPException(status_code=400, detail="Invalid JSON query")
    
    # Parse fields if provided
    projection = None
    if fields:
        projection = {field.strip(): 1 for field in fields.split(",")}
    
    # Get data from the database
    collection_ref = getattr(db, collection)
    cursor = collection_ref.find(query_filter, projection).limit(limit)
    data = await cursor.to_list(length=limit)
    
    # Process data (convert ObjectId, dates, etc.)
    for item in data:
        item["_id"] = str(item["_id"])
        for key, value in item.items():
            if isinstance(value, datetime):
                item[key] = value.isoformat()
            elif isinstance(value, ObjectId):
                item[key] = str(value)
    
    # No data found
    if not data:
        raise HTTPException(status_code=404, detail="No data found matching the criteria")
    
    # Export based on format
    filename = f"{collection}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if format == "json":
        content = json.dumps(data, ensure_ascii=False, indent=2)
        media_type = "application/json"
        filename = f"{filename}.json"

    elif format == "csv":
        output = io.StringIO()
        if data:
            # Get all unique fields across all documents
            all_fields = set()
            for doc in data:
                all_fields.update(doc.keys())
            
            writer = csv.DictWriter(output, fieldnames=sorted(all_fields))
            writer.writeheader()
            for doc in data:
                writer.writerow(doc)
                
        content = output.getvalue()
        media_type = "text/csv"
        filename = f"{filename}.csv"

    elif format == "excel":
        output = io.BytesIO()
        df = pd.DataFrame(data)
        df.to_excel(output, index=False)
        content = output.getvalue()
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"{filename}.xlsx"

    # Return file response
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@router.get("/data",
            summary="Get data for export without downloading",
            description="Returns data that can be used for API submission without forcing a download.",
            response_description="Data for export"
            )
async def get_data_for_export(
        collection: str = Query("articles", description="Data collection to export"),
        query: Optional[str] = Query(None, description="JSON query filter"),
        fields: Optional[str] = Query(None, description="Comma-separated list of fields to include"),
        limit: int = Query(1000, ge=1, le=5000, description="Maximum number of records to export"),
        db=Depends(get_async_db)
):
    """Get data for export without forcing download"""
    # Validate collection name
    valid_collections = ["articles", "urls", "ai_extractions"]
    if collection not in valid_collections:
        raise HTTPException(status_code=400, detail=f"Invalid collection. Must be one of: {', '.join(valid_collections)}")
    
    # Parse query if provided
    query_filter = {}
    if query:
        try:
            query_filter = json.loads(query)
        except:
            raise HTTPException(status_code=400, detail="Invalid JSON query")
    
    # Parse fields if provided
    projection = None
    if fields:
        projection = {field.strip(): 1 for field in fields.split(",")}
    
    # Get data from the database
    collection_ref = getattr(db, collection)
    cursor = collection_ref.find(query_filter, projection).limit(limit)
    data = await cursor.to_list(length=limit)
    
    # Process data (convert ObjectId, dates, etc.)
    for item in data:
        item["_id"] = str(item["_id"])
        for key, value in item.items():
            if isinstance(value, datetime):
                item[key] = value.isoformat()
            elif isinstance(value, ObjectId):
                item[key] = str(value)
    
    return {
        "data": data,
        "count": len(data),
        "collection": collection,
        "timestamp": datetime.now().isoformat()
    }
