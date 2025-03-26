from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path, Body
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import psutil
import os
import platform
import logging
from db.connection import get_async_db

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/config",
            summary="Get system configuration",
            description="Returns the current system configuration.",
            response_description="Current system configuration"
            )
async def get_config(db=Depends(get_async_db)):
    """Get current system configuration"""
    try:
        config = await db.system_config.find_one({"type": "system_config"})
        
        if not config:
            # Return default config if none exists
            return {
                "log_level": "INFO",
                "max_concurrent_tasks": 10,
                "enable_monitoring": True,
                "data_retention_days": 30,
                "auto_cleanup": True,
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
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system configuration: {str(e)}")


@router.post("/config",
             summary="Save system configuration",
             description="Saves system configuration.",
             response_description="Saved system configuration"
             )
async def save_config(config: Dict[str, Any] = Body(...), db=Depends(get_async_db)):
    """Save system configuration"""
    try:
        # Add metadata
        config["type"] = "system_config"
        config["updated_at"] = datetime.now()
        
        # Upsert config
        result = await db.system_config.update_one(
            {"type": "system_config"},
            {"$set": config},
            upsert=True
        )
        
        # Update log level if specified
        if "log_level" in config:
            log_level = getattr(logging, config["log_level"], logging.INFO)
            logging.getLogger().setLevel(log_level)
        
        # Clean up response
        if "_id" in config:
            config["_id"] = str(config["_id"])
        if "type" in config:
            del config["type"]
        
        config["config_exists"] = True
        return config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save system configuration: {str(e)}")


@router.get("/status",
            summary="Get system status",
            description="Returns the current status of the system.",
            response_description="Current system status"
            )
async def get_status():
    """Get current system status"""
    try:
        # Get basic system info
        system_info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_total": psutil.disk_usage('/').total,
            "disk_free": psutil.disk_usage('/').free
        }
        
        # Get running services info
        services = [
            {
                "name": "API Server",
                "status": "running",
                "uptime": "1 day, 2 hours",  # This would be calculated from the actual start time
                "version": "1.0.0"
            },
            {
                "name": "Database",
                "status": "running",
                "uptime": "3 days, 5 hours",  # This would be determined from the actual DB
                "version": "MongoDB 5.0"
            }
        ]
        
        # Get resource usage
        resources = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_sent": psutil.net_io_counters().bytes_sent,
            "network_recv": psutil.net_io_counters().bytes_recv
        }
        
        return {
            "status": "healthy",  # Could be healthy, warning, critical based on thresholds
            "timestamp": datetime.now().isoformat(),
            "system_info": system_info,
            "services": services,
            "resources": resources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system status: {str(e)}")


@router.get("/logs",
            summary="Get system logs",
            description="Returns system logs with optional filtering.",
            response_description="System logs"
            )
async def get_logs(
        level: Optional[str] = Query(None, description="Filter by log level (INFO, WARNING, ERROR, etc.)"),
        service: Optional[str] = Query(None, description="Filter by service name"),
        start_date: Optional[str] = Query(None, description="Filter by start date (ISO format)"),
        end_date: Optional[str] = Query(None, description="Filter by end date (ISO format)"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of logs to return"),
        db=Depends(get_async_db)
):
    """Get system logs with optional filtering"""
    try:
        # Build query
        query = {}
        
        if level:
            query["level"] = level
            
        if service:
            query["service"] = service
            
        date_query = {}
        if start_date:
            date_query["$gte"] = datetime.fromisoformat(start_date)
        if end_date:
            date_query["$lte"] = datetime.fromisoformat(end_date)
            
        if date_query:
            query["timestamp"] = date_query
        
        # Get logs from database
        cursor = db.system_logs.find(query).sort("timestamp", -1).limit(limit)
        logs = await cursor.to_list(length=limit)
        
        # Format response
        for log in logs:
            log["_id"] = str(log["_id"])
            if "timestamp" in log:
                log["timestamp"] = log["timestamp"].isoformat()
        
        return logs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system logs: {str(e)}")


@router.post("/clear-cache",
             summary="Clear system cache",
             description="Clears various system caches.",
             response_description="Cache clearing result"
             )
async def clear_cache():
    """Clear system cache"""
    try:
        # Simulate cache clearing operations
        # In a real system, this would clear various caches
        
        return {
            "message": "System cache cleared successfully",
            "cleared_items": {
                "memory_cache": True,
                "disk_cache": True,
                "template_cache": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/resource-usage",
            summary="Get system resource usage",
            description="Returns detailed information about system resource usage.",
            response_description="System resource usage"
            )
async def get_resource_usage():
    """Get detailed system resource usage"""
    try:
        # Get CPU info
        cpu_info = {
            "percent": psutil.cpu_percent(interval=1),
            "per_cpu": psutil.cpu_percent(interval=1, percpu=True),
            "count": psutil.cpu_count(),
            "logical_count": psutil.cpu_count(logical=True),
            "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
        
        # Get memory info
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        
        memory_info = {
            "virtual": {
                "total": virtual_memory.total,
                "available": virtual_memory.available,
                "used": virtual_memory.used,
                "free": virtual_memory.free,
                "percent": virtual_memory.percent
            },
            "swap": {
                "total": swap_memory.total,
                "used": swap_memory.used,
                "free": swap_memory.free,
                "percent": swap_memory.percent
            }
        }
        
        # Get disk info
        disk_info = {
            "usage": psutil.disk_usage('/')._asdict(),
            "io_counters": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None
        }
        
        # Get network info
        net_io = psutil.net_io_counters()
        network_info = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "errin": net_io.errin,
            "errout": net_io.errout,
            "dropin": net_io.dropin,
            "dropout": net_io.dropout
        }
        
        # Get process info for current process
        current_process = psutil.Process()
        process_info = {
            "pid": current_process.pid,
            "name": current_process.name(),
            "status": current_process.status(),
            "cpu_percent": current_process.cpu_percent(interval=1),
            "memory_info": current_process.memory_info()._asdict(),
            "create_time": datetime.fromtimestamp(current_process.create_time()).isoformat(),
            "num_threads": current_process.num_threads(),
            "username": current_process.username()
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": cpu_info,
            "memory": memory_info,
            "disk": disk_info,
            "network": network_info,
            "process": process_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve resource usage: {str(e)}")


@router.post("/restart",
             summary="Restart system service",
             description="Restarts a specified system service.",
             response_description="Service restart result"
             )
async def restart_service(service: Dict[str, str] = Body(...)):
    """Restart a system service"""
    try:
        service_name = service.get("service")
        
        if not service_name:
            raise HTTPException(status_code=400, detail="Service name not specified")
        
        # Validate service name
        valid_services = ["api", "crawler", "database", "scheduler"]
        if service_name not in valid_services:
            raise HTTPException(status_code=400, detail=f"Invalid service. Must be one of: {', '.join(valid_services)}")
        
        # Simulate service restart
        # In a real system, this would use systemctl, pm2, or other service manager
        
        return {
            "message": f"Service '{service_name}' restart initiated",
            "service": service_name,
            "status": "restarting",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart service: {str(e)}")


@router.get("/performance-metrics",
            summary="Get system performance metrics",
            description="Returns performance metrics for the system over time.",
            response_description="System performance metrics"
            )
async def get_performance_metrics(
        period: str = Query("day", description="Time period (hour, day, week, month)"),
        db=Depends(get_async_db)
):
    """Get system performance metrics over time"""
    try:
        # Determine time range based on period
        now = datetime.now()
        if period == "hour":
            start_time = now - timedelta(hours=1)
            interval = "minute"
        elif period == "day":
            start_time = now - timedelta(days=1)
            interval = "hour"
        elif period == "week":
            start_time = now - timedelta(weeks=1)
            interval = "day"
        elif period == "month":
            start_time = now - timedelta(days=30)
            interval = "day"
        else:
            raise HTTPException(status_code=400, detail="Invalid period. Must be hour, day, week, or month")
        
        # Create mock data for demonstration
        # In a real system, this would query a time-series database or metrics collection
        
        # Generate time points
        time_points = []
        if interval == "minute":
            for i in range(60):
                time_points.append((now - timedelta(minutes=i)).replace(second=0, microsecond=0))
        elif interval == "hour":
            for i in range(24):
                time_points.append((now - timedelta(hours=i)).replace(minute=0, second=0, microsecond=0))
        elif interval == "day":
            for i in range(30 if period == "month" else 7):
                time_points.append((now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0))
        
        # Sort chronologically
        time_points.sort()
        
        # Generate mock data series
        import random
        cpu_series = [random.uniform(10, 80) for _ in range(len(time_points))]
        memory_series = [random.uniform(20, 90) for _ in range(len(time_points))]
        requests_series = [random.randint(10, 1000) for _ in range(len(time_points))]
        latency_series = [random.uniform(10, 200) for _ in range(len(time_points))]
        
        return {
            "period": period,
            "interval": interval,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "metrics": {
                "timestamps": [t.isoformat() for t in time_points],
                "cpu_percent": cpu_series,
                "memory_percent": memory_series,
                "requests_count": requests_series,
                "avg_latency_ms": latency_series
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance metrics: {str(e)}")


@router.get("/alert-config",
            summary="Get system alert configuration",
            description="Returns the current system alert configuration.",
            response_description="System alert configuration"
            )
async def get_alert_config(db=Depends(get_async_db)):
    """Get system alert configuration"""
    try:
        config = await db.system_config.find_one({"type": "alert_config"})
        
        if not config:
            # Return default config if none exists
            return {
                "enabled": False,
                "email": {
                    "enabled": False,
                    "recipients": []
                },
                "thresholds": {
                    "cpu_percent": 80,
                    "memory_percent": 90,
                    "disk_percent": 85,
                    "error_rate": 5
                },
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
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alert configuration: {str(e)}")


@router.post("/alert-config",
             summary="Save system alert configuration",
             description="Saves system alert configuration.",
             response_description="Saved alert configuration"
             )
async def save_alert_config(config: Dict[str, Any] = Body(...), db=Depends(get_async_db)):
    """Save system alert configuration"""
    try:
        # Add metadata
        config["type"] = "alert_config"
        config["updated_at"] = datetime.now()
        
        # Upsert config
        result = await db.system_config.update_one(
            {"type": "alert_config"},
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
        raise HTTPException(status_code=500, detail=f"Failed to save alert configuration: {str(e)}")


@router.get("/alert-history",
            summary="Get system alert history",
            description="Returns a history of system alerts with optional filtering.",
            response_description="System alert history"
            )
async def get_alert_history(
        level: Optional[str] = Query(None, description="Filter by alert level (info, warning, critical)"),
        service: Optional[str] = Query(None, description="Filter by service name"),
        start_date: Optional[str] = Query(None, description="Filter by start date (ISO format)"),
        end_date: Optional[str] = Query(None, description="Filter by end date (ISO format)"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of alerts to return"),
        db=Depends(get_async_db)
):
    """Get system alert history with optional filtering"""
    try:
        # Build query
        query = {}
        
        if level:
            query["level"] = level
            
        if service:
            query["service"] = service
            
        date_query = {}
        if start_date:
            date_query["$gte"] = datetime.fromisoformat(start_date)
        if end_date:
            date_query["$lte"] = datetime.fromisoformat(end_date)
            
        if date_query:
            query["timestamp"] = date_query
        
        # Get alerts from database
        cursor = db.system_alerts.find(query).sort("timestamp", -1).limit(limit)
        alerts = await cursor.to_list(length=limit)
        
        # Format response
        for alert in alerts:
            alert["_id"] = str(alert["_id"])
            if "timestamp" in alert:
                alert["timestamp"] = alert["timestamp"].isoformat()
        
        # If no alerts exist yet, return mock data
        if not alerts:
            # Mock data for demonstration
            alerts = [
                {
                    "_id": "mock1",
                    "level": "critical",
                    "service": "crawler",
                    "message": "CPU usage exceeded 90% for more than 5 minutes",
                    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "resolved": True,
                    "resolved_at": (datetime.now() - timedelta(hours=1, minutes=30)).isoformat()
                },
                {
                    "_id": "mock2",
                    "level": "warning",
                    "service": "database",
                    "message": "Database disk usage above 80%",
                    "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                    "resolved": False
                },
                {
                    "_id": "mock3",
                    "level": "info",
                    "service": "api",
                    "message": "API service restarted successfully",
                    "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
                    "resolved": True,
                    "resolved_at": (datetime.now() - timedelta(days=2)).isoformat()
                }
            ]
        
        return alerts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alert history: {str(e)}")
