import { useState, useEffect, useCallback } from 'react';

/**
 * Custom hook for managing PyTorch model state and API interactions
 */
const usePyTorchModel = () => {
  // State for PyTorch model configuration
  const [usePyTorch, setUsePyTorch] = useState(() => {
    // Restore from localStorage or default to false
    const saved = localStorage.getItem('mlb-dfs-use-pytorch');
    return saved ? JSON.parse(saved) : false;
  });

  const [pytorchBlendWeight, setPytorchBlendWeight] = useState(() => {
    // Restore blend weight or default to 0.5
    const saved = localStorage.getItem('mlb-dfs-pytorch-blend-weight');
    return saved ? parseFloat(saved) : 0.5;
  });

  const [modelStatus, setModelStatus] = useState({
    loading: true,
    available: false,
    trained: false,
    calibrated: false,
    error: null,
    details: null
  });

  // Persist settings to localStorage
  useEffect(() => {
    localStorage.setItem('mlb-dfs-use-pytorch', JSON.stringify(usePyTorch));
  }, [usePyTorch]);

  useEffect(() => {
    localStorage.setItem('mlb-dfs-pytorch-blend-weight', pytorchBlendWeight.toString());
  }, [pytorchBlendWeight]);

  // Check PyTorch model status
  const checkModelStatus = useCallback(async () => {
    setModelStatus(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await fetch('/api/pytorch/model-status');
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        const status = data.status;
        setModelStatus({
          loading: false,
          available: true,
          trained: status.models_available?.all_ready || false,
          calibrated: status.models_available?.calibrators || false,
          error: null,
          details: {
            pa_classifier: status.models_available?.pa_classifier || false,
            stolen_base: status.models_available?.stolen_base || false,
            calibrators: status.models_available?.calibrators || false,
            recommendations: status.recommendations || [],
            training_samples: status.training_status?.estimated_samples || 'Unknown'
          }
        });
      } else {
        throw new Error('Model status check failed');
      }
    } catch (error) {
      console.error('Error checking PyTorch model status:', error);
      setModelStatus({
        loading: false,
        available: false,
        trained: false,
        calibrated: false,
        error: error.message,
        details: null
      });
    }
  }, []);

  // Train PyTorch models
  const trainModels = useCallback(async (options = {}) => {
    const {
      epochs = 50,
      batchSize = 512,
      learningRate = 0.001,
      lookbackDays = 180
    } = options;

    try {
      setModelStatus(prev => ({ ...prev, loading: true, error: null }));

      const response = await fetch('/api/pytorch/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          lookback_days: lookbackDays
        })
      });

      if (!response.ok) {
        throw new Error(`Training failed: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success) {
        // Refresh model status after training
        await checkModelStatus();
        return result;
      } else {
        throw new Error('Training was not successful');
      }
    } catch (error) {
      console.error('Error training PyTorch models:', error);
      setModelStatus(prev => ({
        ...prev,
        loading: false,
        error: error.message
      }));
      throw error;
    }
  }, [checkModelStatus]);

  // Toggle PyTorch usage
  const togglePyTorch = useCallback((enabled) => {
    setUsePyTorch(enabled);
  }, []);

  // Update blend weight
  const updateBlendWeight = useCallback((weight) => {
    const clampedWeight = Math.max(0, Math.min(1, weight));
    setPytorchBlendWeight(clampedWeight);
  }, []);

  // Get display values for UI
  const getDisplayValues = useCallback(() => {
    const isReady = modelStatus.trained && !modelStatus.loading;
    const shouldUsePyTorch = usePyTorch && isReady;
    
    return {
      modelType: shouldUsePyTorch ? 'pytorch' : 'ridge',
      modelLabel: shouldUsePyTorch ? 'PyTorch Enhanced' : 'Ridge Baseline',
      blendPercentage: Math.round(pytorchBlendWeight * 100),
      statusColor: isReady ? (shouldUsePyTorch ? 'indigo' : 'gray') : 'orange',
      statusText: modelStatus.loading 
        ? 'Checking...' 
        : isReady 
          ? (shouldUsePyTorch ? 'Enhanced' : 'Baseline')
          : 'Training Required',
      canUsePyTorch: isReady,
      recommendations: modelStatus.details?.recommendations || []
    };
  }, [usePyTorch, pytorchBlendWeight, modelStatus]);

  // Get API parameters for requests
  const getAPIParams = useCallback(() => {
    const isReady = modelStatus.trained && !modelStatus.loading;
    const shouldUsePyTorch = usePyTorch && isReady;
    
    return {
      use_pytorch: shouldUsePyTorch,
      pytorch_blend_weight: shouldUsePyTorch ? pytorchBlendWeight : 0.0,
      model_type: shouldUsePyTorch ? 'pytorch_enhanced' : 'ridge_baseline'
    };
  }, [usePyTorch, pytorchBlendWeight, modelStatus]);

  // Initialize on mount
  useEffect(() => {
    checkModelStatus();
  }, [checkModelStatus]);

  return {
    // State
    usePyTorch,
    pytorchBlendWeight,
    modelStatus,
    
    // Actions
    togglePyTorch,
    updateBlendWeight,
    checkModelStatus,
    trainModels,
    
    // Computed values
    getDisplayValues,
    getAPIParams,
    
    // Status booleans
    isLoading: modelStatus.loading,
    isReady: modelStatus.trained && !modelStatus.loading,
    hasError: !!modelStatus.error,
    isAvailable: modelStatus.available
  };
};

export default usePyTorchModel;