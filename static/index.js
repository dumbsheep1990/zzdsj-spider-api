import axios from 'axios';

// API基础URL
const API_BASE_URL = 'http://localhost:8000';

// 创建axios实例
const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 30秒超时
    headers: {
        'Content-Type': 'application/json',
    },
});

// 爬虫相关API
export const crawlerAPI = {
    // 获取爬虫状态
    getStatus: () => api.get('/api/crawler/status'),

    // 启动爬虫
    startCrawler: (config) => api.post('/api/crawler/start', config),

    // 通过文件导入URL启动爬虫
    startCrawlerWithFile: (config) => api.post('/api/crawler/start-with-file', config),

    // 停止爬虫
    stopCrawler: () => api.post('/api/crawler/stop'),

    // 获取爬取统计
    getStats: () => api.get('/api/crawler/stats'),

    // 获取子站点列表
    getSubdomains: () => api.get('/api/crawler/subdomains'),

    // 单页爬取
    customCrawl: (formData) => api.post('/api/crawler/custom', formData),

    // 获取单页爬取结果
    getCustomCrawlResult: (crawlId) => api.get(`/api/crawler/custom/${crawlId}`),
    
    // 获取爬虫高级配置 
    getAdvancedConfig: () => api.get('/api/crawler/advanced-config'),
    
    // 保存爬虫高级配置
    saveAdvancedConfig: (config) => api.post('/api/crawler/advanced-config', config)
};

// 文章相关API
export const articleAPI = {
    // 获取文章列表
    getArticles: (params) => api.get('/api/articles', { params }),
    
    // 获取文章详情
    getArticleDetail: (id) => api.get(`/api/article/${id}`),
    
    // 获取文章统计
    getStats: () => api.get('/api/articles/stats'),
    
    // 更新文章状态
    updateStatus: (id, status) => api.put(`/api/article/${id}/status`, { status }),
    
    // 批量更新文章状态
    batchUpdateStatus: (ids, status) => api.put('/api/articles/batch-status', { ids, status }),
    
    // 删除文章
    deleteArticle: (id) => api.delete(`/api/article/${id}`),
    
    // 批量删除文章
    batchDeleteArticles: (ids) => api.delete('/api/articles/batch', { data: { ids } }),
    
    // 导出文章
    exportArticles: (params) => api.get('/api/articles/export', { params }),
    
    // 获取文章分类统计
    getCategoryStats: () => api.get('/api/articles/category-stats'),
    
    // 获取文章标签统计
    getTagStats: () => api.get('/api/articles/tag-stats')
};

// AI提取相关API
export const aiAPI = {
    // 从URL提取内容
    extractFromUrl: (formData) => api.post('/api/ai/extract-url', formData),
    
    // 从HTML内容提取
    extractFromHtml: (formData) => api.post('/api/ai/extract-html', formData),
    
    // 获取提取历史
    getExtractHistory: (params) => api.get('/api/ai/extract-history', { params }),
    
    // 获取提取结果
    getExtractResult: (id) => api.get(`/api/ai/extract-result/${id}`),
    
    // 删除提取历史
    deleteExtractHistory: (id) => api.delete(`/api/ai/extract-history/${id}`),
    
    // 批量删除提取历史
    batchDeleteHistory: (ids) => api.delete('/api/ai/extract-history/batch', { data: { ids } })
};

// 数据清洗相关API
export const cleaningAPI = {
    // 获取清洗配置
    getConfig: () => api.get('/api/cleaning/config'),
    
    // 保存清洗配置
    saveConfig: (config) => api.post('/api/cleaning/config', config),
    
    // 执行数据清洗
    runCleaning: () => api.post('/api/cleaning/run'),
    
    // 获取清洗进度
    getProgress: () => api.get('/api/cleaning/progress'),
    
    // 获取清洗历史
    getHistory: (params) => api.get('/api/cleaning/history', { params }),
    
    // 获取清洗结果统计
    getStats: () => api.get('/api/cleaning/stats'),
    
    // 取消清洗任务
    cancelCleaning: () => api.post('/api/cleaning/cancel'),
    
    // 获取清洗规则列表
    getRules: () => api.get('/api/cleaning/rules'),
    
    // 保存清洗规则
    saveRule: (rule) => api.post('/api/cleaning/rules', rule),
    
    // 删除清洗规则
    deleteRule: (id) => api.delete(`/api/cleaning/rules/${id}`)
};

// 数据导出相关API
export const exportAPI = {
    // 导出数据
    exportData: (params) => api.get('/api/export', { params }),
    
    // 添加获取数据而不下载的方法，用于API提交功能
    getDataForExport: (params) => api.get('/api/export/data', { params })
};

// LLM服务相关API
export const llmAPI = {
    // 获取LLM配置
    getConfig: () => api.get('/api/llm/config'),
    
    // 保存LLM配置
    saveConfig: (config) => api.post('/api/llm/config', config),
    
    // 测试LLM连接
    testConnection: (config) => api.post('/api/llm/test-connection', config),
    
    // 获取可用模型列表
    getAvailableModels: (provider) => api.get(`/api/llm/models/${provider}`),
    
    // 获取模型参数配置
    getModelParams: (model) => api.get(`/api/llm/model-params/${model}`),
    
    // 保存模型参数配置
    saveModelParams: (model, params) => api.post(`/api/llm/model-params/${model}`, params)
};

// 向量化服务相关API
export const vectorAPI = {
    // 获取向量化配置
    getConfig: () => api.get('/api/vector/config'),
    
    // 保存向量化配置
    saveConfig: (config) => api.post('/api/vector/config', config),
    
    // 测试向量化服务连接
    testConnection: (config) => api.post('/api/vector/test-connection', config),
    
    // 获取向量化统计信息
    getStats: () => api.get('/api/vector/stats'),
    
    // 重建向量索引
    rebuildIndex: () => api.post('/api/vector/rebuild-index'),
    
    // 获取向量化进度
    getProgress: () => api.get('/api/vector/progress'),
    
    // 获取向量化历史
    getHistory: (params) => api.get('/api/vector/history', { params }),
    
    // 取消向量化任务
    cancelVectorization: () => api.post('/api/vector/cancel'),
    
    // 获取向量化规则列表
    getRules: () => api.get('/api/vector/rules'),
    
    // 保存向量化规则
    saveRule: (rule) => api.post('/api/vector/rules', rule),
    
    // 删除向量化规则
    deleteRule: (id) => api.delete(`/api/vector/rules/${id}`)
};

// 系统配置相关API
export const systemAPI = {
    // 获取系统配置
    getConfig: () => api.get('/api/system/config'),
    
    // 保存系统配置
    saveConfig: (config) => api.post('/api/system/config', config),
    
    // 获取系统状态
    getStatus: () => api.get('/api/system/status'),
    
    // 获取系统日志
    getLogs: (params) => api.get('/api/system/logs', { params }),
    
    // 清理系统缓存
    clearCache: () => api.post('/api/system/clear-cache'),
    
    // 获取系统资源使用情况
    getResourceUsage: () => api.get('/api/system/resource-usage'),
    
    // 重启系统服务
    restartService: (service) => api.post('/api/system/restart', { service }),
    
    // 获取系统性能指标
    getPerformanceMetrics: () => api.get('/api/system/performance-metrics'),
    
    // 获取系统告警配置
    getAlertConfig: () => api.get('/api/system/alert-config'),
    
    // 保存系统告警配置
    saveAlertConfig: (config) => api.post('/api/system/alert-config', config),
    
    // 获取系统告警历史
    getAlertHistory: (params) => api.get('/api/system/alert-history', { params })
};

export default api;