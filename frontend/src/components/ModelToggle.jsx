import React, { useState, useRef, useEffect } from 'react';
import { 
  Brain, 
  Cpu, 
  Settings, 
  CheckCircle, 
  AlertCircle, 
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Zap,
  BarChart3,
  Info
} from 'lucide-react';
import usePyTorchModel from '../hooks/usePyTorchModel';

const ModelToggle = ({ onModelChange }) => {
  const {
    usePyTorch,
    pytorchBlendWeight,
    modelStatus,
    togglePyTorch,
    updateBlendWeight,
    checkModelStatus,
    trainModels,
    getDisplayValues,
    getAPIParams,
    isLoading,
    isReady,
    hasError,
    isAvailable
  } = usePyTorchModel();

  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);
  const dropdownRef = useRef(null);

  const displayValues = getDisplayValues();

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Notify parent component when model configuration changes
  useEffect(() => {
    if (onModelChange) {
      onModelChange(getAPIParams());
    }
  }, [getAPIParams, onModelChange]);

  const handleTogglePyTorch = (enabled) => {
    togglePyTorch(enabled);
    setIsDropdownOpen(false);
  };

  const handleBlendWeightChange = (event) => {
    const value = parseFloat(event.target.value);
    updateBlendWeight(value);
  };

  const handleTrain = async () => {
    setIsTraining(true);
    try {
      await trainModels({
        epochs: 50,
        batchSize: 512,
        learningRate: 0.001,
        lookbackDays: 180
      });
    } catch (error) {
      console.error('Training failed:', error);
    } finally {
      setIsTraining(false);
    }
  };

  const getStatusIcon = () => {
    if (isLoading || isTraining) {
      return <RefreshCw className="w-4 h-4 animate-spin text-orange-400" />;
    }
    
    if (hasError) {
      return <AlertCircle className="w-4 h-4 text-red-400" />;
    }
    
    if (isReady) {
      return usePyTorch ? 
        <Brain className="w-4 h-4 text-indigo-400" /> : 
        <Cpu className="w-4 h-4 text-gray-400" />;
    }
    
    return <Settings className="w-4 h-4 text-orange-400" />;
  };

  const getStatusColor = () => {
    if (usePyTorch && isReady) {
      return 'from-indigo-500 to-purple-600';
    }
    return 'from-gray-500 to-gray-600';
  };

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Main Toggle Button */}
      <button
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        className={`
          flex items-center space-x-2 px-4 py-2 rounded-lg border transition-all duration-200
          ${usePyTorch && isReady 
            ? 'bg-gradient-to-r from-indigo-500/20 to-purple-600/20 border-indigo-500/30 text-indigo-400' 
            : 'bg-white/5 border-white/10 text-gray-300 hover:bg-white/10'
          }
        `}
        disabled={isLoading}
      >
        {getStatusIcon()}
        <span className="font-medium text-sm">
          Model: {displayValues.modelLabel}
        </span>
        {usePyTorch && isReady && (
          <span className="text-xs bg-indigo-500/30 px-2 py-1 rounded-full">
            {displayValues.blendPercentage}%
          </span>
        )}
        {isDropdownOpen ? 
          <ChevronUp className="w-4 h-4" /> : 
          <ChevronDown className="w-4 h-4" />
        }
      </button>

      {/* Tooltip */}
      {showTooltip && !isDropdownOpen && (
        <div className="absolute top-full left-0 mt-2 p-3 bg-black/90 backdrop-blur-sm rounded-lg border border-white/20 text-sm text-white shadow-xl z-50 min-w-64">
          <div className="flex items-center space-x-2 mb-2">
            <Info className="w-4 h-4 text-indigo-400" />
            <span className="font-semibold">{displayValues.modelLabel}</span>
          </div>
          <p className="text-gray-300 text-xs">
            {usePyTorch && isReady 
              ? `Enhanced PA-level predictions with ${displayValues.blendPercentage}% blend weight. Preserves all correlation logic.`
              : 'Baseline Ridge regression model. Reliable and battle-tested projections.'
            }
          </p>
          {displayValues.recommendations.length > 0 && (
            <div className="mt-2 pt-2 border-t border-white/10">
              <p className="text-xs text-orange-400">
                💡 {displayValues.recommendations[0]}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Dropdown Menu */}
      {isDropdownOpen && (
        <div className="absolute top-full right-0 mt-2 bg-black/90 backdrop-blur-xl rounded-xl border border-white/20 shadow-2xl z-50 min-w-80">
          <div className="p-4 border-b border-white/10">
            <h3 className="text-lg font-bold text-white mb-2 flex items-center">
              <BarChart3 className="w-5 h-5 mr-2 text-indigo-400" />
              Model Configuration
            </h3>
            <p className="text-sm text-gray-400">
              Choose between baseline Ridge regression or enhanced PyTorch models
            </p>
          </div>

          <div className="p-4 space-y-4">
            {/* Model Selection */}
            <div className="space-y-3">
              {/* Ridge Option */}
              <div 
                className={`
                  flex items-center justify-between p-3 rounded-lg border transition-all cursor-pointer
                  ${!usePyTorch 
                    ? 'bg-gray-500/20 border-gray-500/30 text-gray-300' 
                    : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                  }
                `}
                onClick={() => handleTogglePyTorch(false)}
              >
                <div className="flex items-center space-x-3">
                  <Cpu className="w-5 h-5" />
                  <div>
                    <div className="font-medium">Ridge Baseline</div>
                    <div className="text-xs opacity-70">Reliable regression model</div>
                  </div>
                </div>
                {!usePyTorch && <CheckCircle className="w-5 h-5 text-gray-400" />}
              </div>

              {/* PyTorch Option */}
              <div 
                className={`
                  flex items-center justify-between p-3 rounded-lg border transition-all cursor-pointer
                  ${usePyTorch && isReady
                    ? 'bg-gradient-to-r from-indigo-500/20 to-purple-600/20 border-indigo-500/30 text-indigo-300' 
                    : isReady
                      ? 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                      : 'bg-white/5 border-white/10 text-gray-500 cursor-not-allowed'
                  }
                `}
                onClick={() => isReady && handleTogglePyTorch(true)}
              >
                <div className="flex items-center space-x-3">
                  <Brain className="w-5 h-5" />
                  <div>
                    <div className="font-medium flex items-center">
                      PyTorch Enhanced
                      {!isReady && <span className="ml-2 text-xs bg-orange-500/30 px-2 py-1 rounded-full">Training Required</span>}
                    </div>
                    <div className="text-xs opacity-70">PA-level outcome modeling</div>
                  </div>
                </div>
                {usePyTorch && isReady && <CheckCircle className="w-5 h-5 text-indigo-400" />}
              </div>
            </div>

            {/* Blend Weight Slider (only show when PyTorch is selected and ready) */}
            {usePyTorch && isReady && (
              <div className="space-y-3 p-3 bg-indigo-500/10 rounded-lg border border-indigo-500/20">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-indigo-300">
                    Blend Weight
                  </label>
                  <span className="text-sm font-mono bg-indigo-500/30 px-2 py-1 rounded text-indigo-300">
                    {displayValues.blendPercentage}%
                  </span>
                </div>
                <div className="relative">
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={pytorchBlendWeight}
                    onChange={handleBlendWeightChange}
                    className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer slider"
                    style={{
                      background: `linear-gradient(to right, #6366f1 0%, #6366f1 ${displayValues.blendPercentage}%, #ffffff20 ${displayValues.blendPercentage}%, #ffffff20 100%)`
                    }}
                  />
                </div>
                <div className="flex justify-between text-xs text-gray-400">
                  <span>Ridge Only</span>
                  <span>Balanced</span>
                  <span>PyTorch Only</span>
                </div>
                <p className="text-xs text-indigo-400/80">
                  Higher values use more PyTorch predictions while preserving correlation structure.
                </p>
              </div>
            )}

            {/* Training Section (show if models not ready) */}
            {!isReady && isAvailable && (
              <div className="space-y-3 p-3 bg-orange-500/10 rounded-lg border border-orange-500/20">
                <div className="flex items-center space-x-2 text-orange-300">
                  <Zap className="w-4 h-4" />
                  <span className="text-sm font-medium">PyTorch Training Required</span>
                </div>
                <p className="text-xs text-gray-400">
                  Train PyTorch models to enable enhanced PA-level predictions.
                </p>
                <button
                  onClick={handleTrain}
                  disabled={isTraining || isLoading}
                  className="w-full bg-gradient-to-r from-orange-500 to-yellow-500 hover:from-orange-600 hover:to-yellow-600 disabled:opacity-50 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg transition-all text-sm font-medium flex items-center justify-center"
                >
                  {isTraining ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Training Models...
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4 mr-2" />
                      Quick Train (5 min)
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Model Status */}
            <div className="p-3 bg-white/5 rounded-lg">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Status</span>
                <div className="flex items-center space-x-2">
                  {getStatusIcon()}
                  <span className={`
                    ${isReady ? 'text-green-400' : hasError ? 'text-red-400' : 'text-orange-400'}
                  `}>
                    {displayValues.statusText}
                  </span>
                </div>
              </div>
              {modelStatus.details && (
                <div className="mt-2 text-xs text-gray-500 space-y-1">
                  <div>PA Classifier: {modelStatus.details.pa_classifier ? '✓' : '✗'}</div>
                  <div>Stolen Base: {modelStatus.details.stolen_base ? '✓' : '✗'}</div>
                  <div>Calibrators: {modelStatus.details.calibrators ? '✓' : '✗'}</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Custom CSS for slider */}
      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #6366f1;
          cursor: pointer;
          box-shadow: 0 0 8px rgba(99, 102, 241, 0.5);
          transition: all 0.2s ease;
        }
        
        .slider::-webkit-slider-thumb:hover {
          box-shadow: 0 0 12px rgba(99, 102, 241, 0.8);
          transform: scale(1.1);
        }
        
        .slider::-moz-range-thumb {
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #6366f1;
          cursor: pointer;
          border: none;
          box-shadow: 0 0 8px rgba(99, 102, 241, 0.5);
          transition: all 0.2s ease;
        }
        
        .slider::-moz-range-thumb:hover {
          box-shadow: 0 0 12px rgba(99, 102, 241, 0.8);
          transform: scale(1.1);
        }
      `}</style>
    </div>
  );
};

export default ModelToggle;