import React, { useState, useEffect } from 'react';
import { 
  Globe, 
  Download, 
  Search, 
  AlertCircle, 
  CheckCircle, 
  Loader2,
  ExternalLink,
  FileText,
  Activity,
  Cloud,
  Zap,
  Settings,
  Copy,
  Play,
  Pause,
  RotateCcw,
  Cpu,
  Clock
} from 'lucide-react';

const FirecrawlPanel = () => {
  const [activeTab, setActiveTab] = useState('scrape');
  const [url, setUrl] = useState('');
  const [urls, setUrls] = useState(['']);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  // Extract-specific states
  const [extractPrompt, setExtractPrompt] = useState('');
  const [extractSchema, setExtractSchema] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [enableWebSearch, setEnableWebSearch] = useState(false);
  const [asyncMode, setAsyncMode] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [templates, setTemplates] = useState({});
  const [schemas, setSchemas] = useState({});
  const [prompts, setPrompts] = useState({});

  // Load schemas and prompts on component mount
  useEffect(() => {
    const loadTemplates = async () => {
      try {
        const [schemasRes, promptsRes] = await Promise.all([
          fetch('/api/extract/schemas'),
          fetch('/api/extract/prompts')
        ]);
        
        if (schemasRes.ok && promptsRes.ok) {
          const schemasData = await schemasRes.json();
          const promptsData = await promptsRes.json();
          setSchemas(schemasData.schemas);
          setPrompts(promptsData.prompts);
          
          // Create templates combining schemas and prompts
          const templateOptions = {};
          Object.keys(schemasData.schemas).forEach(key => {
            templateOptions[key] = {
              name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
              schema: schemasData.schemas[key],
              prompt: promptsData.prompts[key]
            };
          });
          setTemplates(templateOptions);
        }
      } catch (err) {
        console.error('Failed to load templates:', err);
      }
    };
    
    loadTemplates();
  }, []);

  // Poll job status for async extractions
  useEffect(() => {
    let interval;
    if (jobId && (jobStatus === 'processing' || jobStatus === null)) {
      interval = setInterval(async () => {
        try {
          const response = await fetch(`/api/extract/status/${jobId}`);
          if (response.ok) {
            const data = await response.json();
            setJobStatus(data.status);
            if (data.status === 'completed') {
              setResult(data);
              setLoading(false);
            } else if (data.status === 'failed') {
              setError('Extraction job failed');
              setLoading(false);
            }
          }
        } catch (err) {
          console.error('Error checking job status:', err);
        }
      }, 2000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jobId, jobStatus]);

  const handleScrape = async (endpoint = '/api/scrape', payload = {}) => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to scrape data');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSingleScrape = () => {
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }
    handleScrape('/api/scrape', { url: url.trim() });
  };

  const handleMLBStatsScrape = () => {
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }
    handleScrape('/api/scrape/mlb-stats', { url: url.trim() });
  };

  const handleInjuryReportScrape = () => {
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }
    handleScrape('/api/scrape/injury-report', { url: url.trim() });
  };

  const handleWeatherScrape = () => {
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }
    handleScrape('/api/scrape/weather', { url: url.trim() });
  };

  const handleBulkScrape = () => {
    const validUrls = urls.filter(u => u.trim());
    if (validUrls.length === 0) {
      setError('Please enter at least one URL');
      return;
    }
    handleScrape('/api/scrape/bulk', { urls: validUrls });
  };

  const handleCrawl = () => {
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }
    handleScrape('/api/crawl', { url: url.trim() });
  };

  const addUrlField = () => {
    setUrls([...urls, '']);
  };

  const updateUrl = (index, value) => {
    const newUrls = [...urls];
    newUrls[index] = value;
    setUrls(newUrls);
  };

  const removeUrl = (index) => {
    if (urls.length > 1) {
      setUrls(urls.filter((_, i) => i !== index));
    }
  };

  // Extract-specific handlers
  const handleExtract = async () => {
    const validUrls = activeTab === 'extract' ? urls.filter(u => u.trim()) : [url].filter(u => u.trim());
    if (validUrls.length === 0) {
      setError('Please enter at least one URL');
      return;
    }

    const payload = {
      urls: validUrls,
      enableWebSearch
    };

    if (extractPrompt) payload.prompt = extractPrompt;
    if (extractSchema) {
      try {
        payload.schema = JSON.parse(extractSchema);
      } catch (e) {
        setError('Invalid JSON schema format');
        return;
      }
    }

    if (!payload.prompt && !payload.schema) {
      setError('Please provide either a prompt or schema');
      return;
    }

    const endpoint = asyncMode ? '/api/extract/start' : '/api/extract';
    const result = await handleScrape(endpoint, payload);
    
    if (asyncMode && result && result.job_id) {
      setJobId(result.job_id);
      setJobStatus('processing');
    }
  };

  const handleTemplateExtract = async () => {
    if (!selectedTemplate) {
      setError('Please select a template');
      return;
    }

    const validUrls = urls.filter(u => u.trim());
    if (validUrls.length === 0) {
      setError('Please enter at least one URL');
      return;
    }

    const payload = {
      template: selectedTemplate,
      urls: validUrls,
      enableWebSearch
    };

    handleScrape('/api/extract/template', payload);
  };

  const handleMLBSpecificExtract = async (type) => {
    const validUrls = activeTab === 'extract' ? urls.filter(u => u.trim()) : [url].filter(u => u.trim());
    if (validUrls.length === 0) {
      setError('Please enter at least one URL');
      return;
    }

    const endpoints = {
      'season-stats': '/api/extract/mlb-season-stats',
      'dfs-slate': '/api/extract/dfs-slate',
      'betting-lines': '/api/extract/betting-lines',
      'injury-report': '/api/extract/advanced-injury-report'
    };

    const payload = type === 'season-stats' 
      ? { url: validUrls[0], enableWebSearch }
      : { urls: validUrls };

    handleScrape(endpoints[type], payload);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  const loadTemplate = (templateKey) => {
    if (templates[templateKey]) {
      setExtractPrompt(templates[templateKey].prompt);
      setExtractSchema(JSON.stringify(templates[templateKey].schema, null, 2));
      setSelectedTemplate(templateKey);
    }
  };

  const formatResult = (data) => {
    if (!data) return null;
    
    try {
      return JSON.stringify(data, null, 2);
    } catch {
      return String(data);
    }
  };

  return (
    <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-6">
      <div className="flex items-center mb-6">
        <Globe className="w-6 h-6 text-blue-400 mr-3" />
        <h2 className="text-xl font-bold text-white">Web Data Collection</h2>
        <span className="ml-2 px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded-full">
          Powered by Firecrawl
        </span>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 mb-6 bg-white/5 rounded-lg p-1">
        {[
          { id: 'scrape', label: 'Single Scrape', icon: FileText },
          { id: 'bulk', label: 'Bulk Scrape', icon: Activity },
          { id: 'crawl', label: 'Site Crawl', icon: Cloud },
          { id: 'extract', label: 'AI Extract', icon: Zap },
          { id: 'templates', label: 'MLB Templates', icon: Cpu }
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
              activeTab === id
                ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <Icon className="w-4 h-4" />
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* Single Scrape Tab */}
      {activeTab === 'scrape' && (
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Website URL
            </label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/page"
              className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <button
              onClick={handleSingleScrape}
              disabled={loading}
              className="flex items-center justify-center space-x-2 px-4 py-3 bg-blue-500 hover:bg-blue-600 disabled:opacity-50 text-white rounded-lg transition-colors"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              <span>General Scrape</span>
            </button>

            <button
              onClick={handleMLBStatsScrape}
              disabled={loading}
              className="flex items-center justify-center space-x-2 px-4 py-3 bg-green-500 hover:bg-green-600 disabled:opacity-50 text-white rounded-lg transition-colors"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
              <span>MLB Stats</span>
            </button>

            <button
              onClick={handleInjuryReportScrape}
              disabled={loading}
              className="flex items-center justify-center space-x-2 px-4 py-3 bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white rounded-lg transition-colors"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <AlertCircle className="w-4 h-4" />}
              <span>Injury Report</span>
            </button>

            <button
              onClick={handleWeatherScrape}
              disabled={loading}
              className="flex items-center justify-center space-x-2 px-4 py-3 bg-purple-500 hover:bg-purple-600 disabled:opacity-50 text-white rounded-lg transition-colors"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Cloud className="w-4 h-4" />}
              <span>Weather Data</span>
            </button>
          </div>
        </div>
      )}

      {/* Bulk Scrape Tab */}
      {activeTab === 'bulk' && (
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              URLs to Scrape
            </label>
            <div className="space-y-3">
              {urls.map((url, index) => (
                <div key={index} className="flex space-x-2">
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => updateUrl(index, e.target.value)}
                    placeholder={`https://example.com/page-${index + 1}`}
                    className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                  />
                  {urls.length > 1 && (
                    <button
                      onClick={() => removeUrl(index)}
                      className="px-3 py-3 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                    >
                      ×
                    </button>
                  )}
                </div>
              ))}
            </div>
            <button
              onClick={addUrlField}
              className="mt-3 px-4 py-2 bg-white/5 text-gray-300 rounded-lg hover:bg-white/10 transition-colors"
            >
              + Add URL
            </button>
          </div>

          <button
            onClick={handleBulkScrape}
            disabled={loading}
            className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-blue-500 hover:bg-blue-600 disabled:opacity-50 text-white rounded-lg transition-colors"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            <span>Scrape All URLs</span>
          </button>
        </div>
      )}

      {/* Crawl Tab */}
      {activeTab === 'crawl' && (
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Starting URL
            </label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com"
              className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
            <p className="mt-2 text-sm text-gray-400">
              This will crawl the entire website starting from the provided URL
            </p>
          </div>

          <button
            onClick={handleCrawl}
            disabled={loading}
            className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-purple-500 hover:bg-purple-600 disabled:opacity-50 text-white rounded-lg transition-colors"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Globe className="w-4 h-4" />}
            <span>Start Crawl</span>
          </button>
        </div>
      )}

      {/* AI Extract Tab */}
      {activeTab === 'extract' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* URLs Section */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                URLs to Extract From
              </label>
              <div className="space-y-3">
                {urls.map((url, index) => (
                  <div key={index} className="flex space-x-2">
                    <input
                      type="url"
                      value={url}
                      onChange={(e) => updateUrl(index, e.target.value)}
                      placeholder={`https://example.com/page-${index + 1}`}
                      className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                    />
                    {urls.length > 1 && (
                      <button
                        onClick={() => removeUrl(index)}
                        className="px-3 py-3 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                      >
                        ×
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <button
                onClick={addUrlField}
                className="mt-3 px-4 py-2 bg-white/5 text-gray-300 rounded-lg hover:bg-white/10 transition-colors"
              >
                + Add URL
              </button>
              <p className="mt-2 text-sm text-gray-400">
                💡 Use wildcards like https://site.com/* to extract from entire domains
              </p>
            </div>

            {/* Settings Section */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-4">
                <Settings className="w-4 h-4 inline mr-2" />
                Extraction Settings
              </label>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Enable Web Search</span>
                  <button
                    onClick={() => setEnableWebSearch(!enableWebSearch)}
                    className={`w-12 h-6 rounded-full transition-colors ${
                      enableWebSearch ? 'bg-blue-500' : 'bg-gray-600'
                    }`}
                  >
                    <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
                      enableWebSearch ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Async Mode</span>
                  <button
                    onClick={() => setAsyncMode(!asyncMode)}
                    className={`w-12 h-6 rounded-full transition-colors ${
                      asyncMode ? 'bg-purple-500' : 'bg-gray-600'
                    }`}
                  >
                    <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
                      asyncMode ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
              </div>
              
              {enableWebSearch && (
                <div className="mt-3 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <p className="text-sm text-blue-200">
                    Web search will follow links to gather additional context and comprehensive data.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Prompt Section */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Extraction Prompt
            </label>
            <textarea
              value={extractPrompt}
              onChange={(e) => setExtractPrompt(e.target.value)}
              placeholder="Describe what data you want to extract... e.g., 'Extract player stats including batting average, home runs, and RBI'"
              rows={4}
              className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Schema Section */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-300">
                JSON Schema (Optional)
              </label>
              <button
                onClick={() => copyToClipboard(extractSchema)}
                className="px-3 py-1 bg-white/5 text-gray-400 rounded hover:bg-white/10 transition-colors"
              >
                <Copy className="w-4 h-4" />
              </button>
            </div>
            <textarea
              value={extractSchema}
              onChange={(e) => setExtractSchema(e.target.value)}
              placeholder='{"type": "object", "properties": {"players": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "stats": {"type": "object"}}}}}}'
              rows={6}
              className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 font-mono text-sm"
            />
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-4">
            <button
              onClick={handleExtract}
              disabled={loading}
              className="flex-1 flex items-center justify-center space-x-2 px-4 py-3 bg-blue-500 hover:bg-blue-600 disabled:opacity-50 text-white rounded-lg transition-colors"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              <span>{asyncMode ? 'Start Extract Job' : 'Extract Data'}</span>
            </button>

            <button
              onClick={() => {
                setExtractPrompt('');
                setExtractSchema('');
                setSelectedTemplate('');
                setResult(null);
                setError(null);
              }}
              className="px-4 py-3 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {/* Quick MLB Extracts */}
          <div className="border-t border-white/10 pt-6">
            <h4 className="text-white font-medium mb-4">Quick MLB Extracts</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <button
                onClick={() => handleMLBSpecificExtract('season-stats')}
                disabled={loading}
                className="flex items-center justify-center space-x-2 px-4 py-3 bg-green-500 hover:bg-green-600 disabled:opacity-50 text-white rounded-lg transition-colors"
              >
                <Activity className="w-4 h-4" />
                <span>Season Stats</span>
              </button>

              <button
                onClick={() => handleMLBSpecificExtract('dfs-slate')}
                disabled={loading}
                className="flex items-center justify-center space-x-2 px-4 py-3 bg-purple-500 hover:bg-purple-600 disabled:opacity-50 text-white rounded-lg transition-colors"
              >
                <FileText className="w-4 h-4" />
                <span>DFS Slate</span>
              </button>

              <button
                onClick={() => handleMLBSpecificExtract('betting-lines')}
                disabled={loading}
                className="flex items-center justify-center space-x-2 px-4 py-3 bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white rounded-lg transition-colors"
              >
                <Cpu className="w-4 h-4" />
                <span>Betting Lines</span>
              </button>

              <button
                onClick={() => handleMLBSpecificExtract('injury-report')}
                disabled={loading}
                className="flex items-center justify-center space-x-2 px-4 py-3 bg-red-500 hover:bg-red-600 disabled:opacity-50 text-white rounded-lg transition-colors"
              >
                <AlertCircle className="w-4 h-4" />
                <span>Injury Report</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MLB Templates Tab */}
      {activeTab === 'templates' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(templates).map(([key, template]) => (
              <div key={key} className="bg-white/5 border border-white/10 rounded-lg p-4 hover:border-white/20 transition-all">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-white font-medium">{template.name}</h4>
                  <Cpu className="w-5 h-5 text-blue-400" />
                </div>
                <p className="text-sm text-gray-400 mb-4">
                  {template.prompt.substring(0, 100)}...
                </p>
                <div className="flex space-x-2">
                  <button
                    onClick={() => loadTemplate(key)}
                    className="flex-1 px-3 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded text-sm transition-colors"
                  >
                    Load Template
                  </button>
                  <button
                    onClick={() => {
                      setSelectedTemplate(key);
                      handleTemplateExtract();
                    }}
                    disabled={loading || urls.filter(u => u.trim()).length === 0}
                    className="px-3 py-2 bg-green-500 hover:bg-green-600 disabled:opacity-50 text-white rounded text-sm transition-colors"
                  >
                    <Play className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>

          {/* Template URLs */}
          <div className="border-t border-white/10 pt-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              URLs for Template Extraction
            </label>
            <div className="space-y-3">
              {urls.map((url, index) => (
                <div key={index} className="flex space-x-2">
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => updateUrl(index, e.target.value)}
                    placeholder={`https://baseball-reference.com/leagues/MLB/2024-standard-batting.shtml`}
                    className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                  />
                  {urls.length > 1 && (
                    <button
                      onClick={() => removeUrl(index)}
                      className="px-3 py-3 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                    >
                      ×
                    </button>
                  )}
                </div>
              ))}
            </div>
            <button
              onClick={addUrlField}
              className="mt-3 px-4 py-2 bg-white/5 text-gray-300 rounded-lg hover:bg-white/10 transition-colors"
            >
              + Add URL
            </button>
          </div>
        </div>
      )}

      {/* Job Status Display */}
      {jobId && jobStatus && (
        <div className="mt-6 bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <Clock className="w-5 h-5 text-purple-400" />
              <span className="text-purple-200 font-medium">Extraction Job Status</span>
            </div>
            <div className={`px-3 py-1 rounded-full text-sm ${
              jobStatus === 'completed' ? 'bg-green-500/20 text-green-400' :
              jobStatus === 'processing' ? 'bg-yellow-500/20 text-yellow-400' :
              jobStatus === 'failed' ? 'bg-red-500/20 text-red-400' :
              'bg-gray-500/20 text-gray-400'
            }`}>
              {jobStatus}
            </div>
          </div>
          <div className="mt-2 text-sm text-purple-200/80">
            Job ID: {jobId}
          </div>
          {jobStatus === 'processing' && (
            <div className="mt-3 flex items-center space-x-2">
              <Loader2 className="w-4 h-4 animate-spin text-purple-400" />
              <span className="text-sm text-purple-200">Processing... This may take a few minutes for large extractions.</span>
            </div>
          )}
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mt-6 flex items-start space-x-3 bg-red-500/10 border border-red-500/30 text-red-200 px-4 py-3 rounded-lg">
          <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-semibold">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* Success Display */}
      {result && (
        <div className="mt-6 space-y-4">
          <div className="flex items-center space-x-3 bg-green-500/10 border border-green-500/30 text-green-200 px-4 py-3 rounded-lg">
            <CheckCircle className="w-5 h-5 flex-shrink-0" />
            <div>
              <p className="font-semibold">Success</p>
              <p className="text-sm">Data scraped successfully</p>
              {result.url && (
                <a 
                  href={result.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-1 text-sm text-green-300 hover:text-green-200 mt-1"
                >
                  <span>Source: {result.url}</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </div>
          </div>

          {/* Result Preview */}
          <div className="bg-white/5 border border-white/10 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-white font-medium">Scraped Data</h4>
              <span className="text-xs text-gray-400">
                {result.type && `Type: ${result.type}`}
              </span>
            </div>
            <div className="max-h-96 overflow-y-auto">
              <pre className="text-sm text-gray-300 whitespace-pre-wrap">
                {formatResult(result.data)}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-6 bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <h4 className="text-blue-200 font-medium mb-2">How to Use</h4>
        <ul className="text-sm text-blue-200/80 space-y-1">
          <li>• <strong>General Scrape:</strong> Extract any webpage content</li>
          <li>• <strong>MLB Stats:</strong> Optimized for baseball statistics sites</li>
          <li>• <strong>Injury Report:</strong> Structured extraction of injury data</li>
          <li>• <strong>Weather Data:</strong> Game weather conditions</li>
          <li>• <strong>Bulk Scrape:</strong> Process multiple URLs at once</li>
          <li>• <strong>Site Crawl:</strong> Discover and scrape entire websites</li>
          <li>• <strong>AI Extract:</strong> Use LLMs to intelligently extract structured data</li>
          <li>• <strong>MLB Templates:</strong> Pre-built schemas for common MLB data types</li>
        </ul>
      </div>
    </div>
  );
};

export default FirecrawlPanel;