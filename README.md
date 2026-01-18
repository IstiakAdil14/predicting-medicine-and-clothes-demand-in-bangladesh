# 🇧🇩 Bangladesh Demand Prediction System

A comprehensive machine learning system that accurately predicts demand for clothes and medicines across Bangladesh's major cities using Random Forest algorithms, achieving 85% accuracy for clothes and 79% for medicine predictions.

## 🎯 Project Overview

This system uses Random Forest Regressor models to predict demand for 6 clothing items and 6 medicine types across 10 major areas in Bangladesh, covering the period from 2010-2025 with high accuracy rates.

## 🌟 Key Features

### 📱 Web Application
- **Interactive Interface**: Modern HTML/CSS/JS web application with Chart.js visualizations
- **Dual Prediction System**: Separate models for clothes and medicine demand
- **Real-time Visualization**: Dynamic charts showing actual vs predicted trends
- **User-friendly Design**: Responsive interface with loading animations and error handling

### 🤖 Machine Learning Models
- **Algorithm**: Random Forest Regressor (300 estimators, max_depth=10)
- **Accuracy**: 85% R² for clothes, 79% R² for medicine
- **Features**: 10 features each (population, density, year_scaled, pop_density_per_1000 + 6 lag features)
- **Validation**: GroupKFold cross-validation by district to prevent geographical data leakage
- **Error Metrics**: MAE ~15 units (clothes), ~12 units (medicine)

## 📊 Prediction Capabilities

### 👕 Clothes Items (6 types)
- **Shirts**, **Pants**, **Jackets**, **Sarees**, **Dresses**, **Coats**

### 💊 Medicine Types (6 types)
- **Antibiotics**, **Painkillers**, **Antacids**, **Vitamins**, **Antihistamines**, **Insulin**

### 🏙️ Coverage Areas (10 locations)
- **Dhaka**: North, South, Gazipur, Narsingdi
- **Chittagong**: City, Cox's Bazar, Feni, Comilla  
- **Khulna**: City, Jessore, Satkhira, Bagerhat
- **Rajshahi**: City, Pabna, Natore, Bogra
- **Barisal**: City, Patuakhali, Bhola, Jhalokathi
- **Sylhet**: City, Moulvibazar, Habiganj, Sunamganj
- **Rangpur**: City, Dinajpur, Thakurgaon, Lalmonirhat
- **Mymensingh**: City, Netrokona, Jamalpur, Sherpur

## 🚀 Quick Start Guide

### Web Application
1. Open `index.html` in your browser
2. Select **Clothes** or **Medicine** tab
3. Choose **Year** (2010-2030), **Item**, and **Area**
4. Click **"🔮 Predict Demand"** for single prediction
5. Click **"📈 Show Trend"** for historical trend analysis

### Python Models
```bash
# Data preprocessing (optional)
cd clothes-demand-prediction
python build_clothes_demand_csv.py

# Train and test clothes model
python test_and_train.py

# Medicine model
cd ../medicine-demand-prediction
python build_medicine_demand_csv.py
python test_and_train.py
```

## 📁 Project Structure

```
predicting-medicine-and-clothes-demand-in-Bangladesh/
├── index.html                          # Main web application
├── style.css                           # Web styling
├── script.js                           # Frontend JavaScript with Chart.js
├── clothes-demand-prediction/
│   ├── bangladesh_clothes_demand.csv   # Raw training data (2010-2025)
│   ├── cleaned_clothes_demand.csv      # Processed data
│   ├── test_and_train.py              # ML model training & prediction interface
│   └── build_clothes_demand_csv.py     # Data preprocessing pipeline
├── medicine-demand-prediction/
│   ├── bangladesh_medicine_demand.csv  # Raw training data (2010-2025)
│   ├── cleaned_medicine_demand.csv     # Processed data
│   ├── test_and_train.py              # ML model training & prediction interface
│   └── build_medicine_demand_csv.py    # Data preprocessing pipeline
├── Lab_Viva_QA_Complete.md            # 200 Q&A for lab viva preparation
├── FUTURE_PREDICTION_GUIDE.md         # Future enhancement guide
├── PROJECT_OVERVIEW.md                # Detailed project documentation
└── README.md                           # This file
```

## 🔧 Technical Implementation

### Dependencies
```python
# Core ML Libraries
pandas>=1.3.0          # Data manipulation and analysis
numpy>=1.21.0           # Numerical computing
scikit-learn>=1.0.0     # Machine learning algorithms
matplotlib>=3.5.0       # Data visualization

# Web Technologies
# Chart.js (CDN)         # Interactive charts
# HTML5/CSS3/ES6        # Frontend technologies
```

### Model Architecture & Performance

#### Random Forest Configuration
```python
RandomForestRegressor(
    n_estimators=300,      # Number of trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=3,    # Minimum samples per leaf
    max_features="sqrt",   # Features per split
    random_state=42        # Reproducibility
)
```

#### Performance Metrics
- **Clothes Model**: R² = 0.85, MAE = 15.2 units, RMSE = 18.7 units
- **Medicine Model**: R² = 0.79, MAE = 12.4 units, RMSE = 16.1 units
- **Training Time**: ~3-5 seconds per model
- **Prediction Time**: <0.1 seconds
- **Cross-validation**: GroupKFold by district (prevents data leakage)

### Feature Engineering

#### Input Features (10 total)
1. **population** - Raw population count
2. **pop_density** - Population per km²
3. **pop_density_per_1000** - Scaled density feature
4. **year_scaled** - Min-max normalized year (0-1)
5. **{item}_prev** - Previous year demand (6 lag features)

#### Data Preprocessing Pipeline
1. **Missing Value Imputation**: Hierarchical median imputation
2. **Outlier Handling**: IQR-based clipping (Q1-1.5*IQR, Q3+1.5*IQR)
3. **Feature Scaling**: Min-max normalization for temporal features
4. **Lag Feature Creation**: Previous year demand using pandas shift()
5. **Data Validation**: Type conversion and consistency checks

## 📈 Model Validation & Metrics

### Evaluation Strategy
- **Cross-Validation**: GroupKFold by district (prevents geographical data leakage)
- **Train-Test Split**: Per-district validation for geographical generalization
- **Temporal Validation**: Models tested across 2010-2025 time range

### Performance Metrics Explained
- **R² Score**: Coefficient of determination (0-1, higher = better variance explanation)
- **MAE**: Mean Absolute Error in demand units (lower = better accuracy)
- **RMSE**: Root Mean Squared Error (penalizes larger errors more)
- **MSE**: Mean Squared Error (base error metric)

### District-wise Performance
- **Consistent Accuracy**: Models perform reliably across all 8 divisions
- **Urban vs Rural**: Higher accuracy in urban areas due to more stable patterns
- **Temporal Stability**: Performance maintained across 15-year period

## 🎨 Web Application Features

### Frontend Technologies
- **HTML5**: Semantic markup with modern standards
- **CSS3**: Responsive design with flexbox/grid layouts
- **JavaScript ES6**: Modern JS with async/await patterns
- **Chart.js**: Interactive line charts for trend visualization

### User Experience
- **Responsive Design**: Mobile-first approach, works on all devices
- **Loading States**: Smooth animations during prediction processing
- **Error Handling**: Comprehensive validation with user-friendly messages
- **Accessibility**: ARIA labels and keyboard navigation support
- **Performance**: Optimized for fast loading and smooth interactions

## 💡 Real-World Applications

### 🏪 For Retailers & Fashion Industry
- **Inventory Management**: Optimize stock levels to prevent overstocking/understocking
- **Seasonal Planning**: Predict demand spikes for festivals (Eid, Durga Puja)
- **Regional Strategy**: Tailor product mix based on area-specific preferences
- **Supply Chain**: Coordinate with manufacturers based on demand forecasts
- **Pricing Strategy**: Dynamic pricing based on predicted demand patterns

### 🏥 For Healthcare & Pharmaceutical Sector
- **Medicine Inventory**: Ensure adequate supply of essential medications
- **Emergency Preparedness**: Stock critical medicines during disease outbreaks
- **Waste Reduction**: Minimize expired medicine losses through accurate forecasting
- **Distribution Planning**: Optimize medicine distribution across regions
- **Budget Planning**: Allocate healthcare budgets based on predicted needs

### 🏛️ For Government & Policy Makers
- **Healthcare Policy**: Data-driven decisions for public health initiatives
- **Economic Planning**: Understand market trends in textile and pharmaceutical sectors
- **Resource Allocation**: Distribute government supplies efficiently across districts
- **Import Planning**: Forecast import needs for medicines and raw materials
- **Development Programs**: Target development programs based on demand patterns

### 📊 For Research & Analytics
- **Market Research**: Understand consumer behavior patterns
- **Academic Studies**: Demographic impact on consumption patterns
- **Business Intelligence**: Strategic planning for market expansion
- **Trend Analysis**: Long-term market trend identification

## 🔮 Future Enhancements

### 🚀 Planned Features
- **Real-time Data Integration**: Connect to live demographic and economic data APIs
- **Advanced Models**: Deep learning models (LSTM, GRU) for time series forecasting
- **Seasonal Intelligence**: Weather data, cultural events, and festival impact analysis
- **Mobile Application**: React Native app for on-the-go predictions
- **API Development**: RESTful API with authentication for third-party integration
- **Dashboard Analytics**: Advanced business intelligence dashboard
- **Multi-language Support**: Bengali and English language options

### 📈 Model Improvements
- **Ensemble Methods**: Combine multiple algorithms for better accuracy
- **Feature Engineering**: Economic indicators, weather data, social media trends
- **Hyperparameter Optimization**: Automated tuning using GridSearchCV/RandomizedSearchCV
- **Online Learning**: Models that update with new data automatically

## 🛠️ Development Setup

### Prerequisites
```bash
# Python 3.8+
pip install pandas numpy scikit-learn matplotlib

# For web development
# Modern web browser with JavaScript enabled
```

### Installation
```bash
git clone https://github.com/yourusername/bangladesh-demand-prediction.git
cd bangladesh-demand-prediction

# Run preprocessing (optional)
cd clothes-demand-prediction
python build_clothes_demand_csv.py

# Train models
python test_and_train.py
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/NewFeature`)
3. **Commit** your changes (`git commit -m 'Add NewFeature'`)
4. **Push** to the branch (`git push origin feature/NewFeature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 for Python code
- Add unit tests for new features
- Update documentation for API changes
- Ensure cross-platform compatibility

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👥 Team

**Project Maintainer**
- GitHub: [@istiakadil14](https://github.com/IstiakAdil14/)
- Email: istiakadil346@gmail.com

## 🙏 Acknowledgments

- **Bangladesh Bureau of Statistics** for demographic data insights
- **Scikit-learn Community** for robust machine learning tools
- **Chart.js Team** for beautiful data visualizations
- **Open Source Community** for continuous inspiration and support

## 📚 Additional Resources

- [Lab Viva Q&A](Lab_Viva_QA_Complete.md) - 200 questions for exam preparation
- [Future Prediction Guide](FUTURE_PREDICTION_GUIDE.md) - Advanced prediction techniques
- [Project Overview](PROJECT_OVERVIEW.md) - Detailed technical documentation

---

**⭐ Star this repository if you found it helpful!**

**🔗 Share with your network to help others learn ML applications in Bangladesh!**
