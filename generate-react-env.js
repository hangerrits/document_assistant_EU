const fs = require('fs');
const path = require('path');

// Create/clear the log file
const logFilePath = path.resolve(__dirname, 'env-generation.log');
fs.writeFileSync(logFilePath, '', 'utf8');

// Function to log both to console and to the log file
function log(message) {
  console.log(message);
  fs.appendFileSync(logFilePath, message + '\n', 'utf8');
}

log('🔄 Starting environment variable generation for React...');

// Path to the backend .env file (adjust if needed)
const backendEnvPath = path.resolve(__dirname, '.env');
log(`📂 Looking for backend .env file at: ${backendEnvPath}`);

// Path where we'll create the React .env.local file
const reactEnvPath = path.resolve(__dirname, '.env.local');
log(`📝 Will create React .env.local file at: ${reactEnvPath}`);

// Read the backend .env file
try {
  log('📖 Reading backend .env file...');
  
  if (!fs.existsSync(backendEnvPath)) {
    throw new Error(`Backend .env file not found at ${backendEnvPath}`);
  }
  
  const envContent = fs.readFileSync(backendEnvPath, 'utf8');
  log('✅ Successfully read .env file');
  
  // Parse the .env content
  const envLines = envContent.split('\n');
  const reactEnvLines = [];
  const processedVars = [];
  
  log('🔍 Processing environment variables...');
  
  // Process each line
  envLines.forEach(line => {
    // Skip empty lines and comments
    if (!line.trim() || line.startsWith('#')) {
      reactEnvLines.push(line);
      return;
    }
    
    // Extract variable name and value
    const match = line.match(/^([^=]+)=(.*)$/);
    if (match) {
      const [, name, value] = match;
      
      // Process specific backend variables that need to be exposed to React
      if (name === 'BACKEND_URL') {
        reactEnvLines.push(`REACT_APP_${name}=${value}`);
        processedVars.push({ 
          original: name, 
          react: `REACT_APP_${name}`, 
          value: value 
        });
      }
      
      // You can add more variables here as needed:
      // if (name === 'ANOTHER_VARIABLE') {
      //   reactEnvLines.push(`REACT_APP_${name}=${value}`);
      //   processedVars.push({ original: name, react: `REACT_APP_${name}`, value: value });
      // }
    }
  });
  
  // Write the React .env.local file
  fs.writeFileSync(reactEnvPath, reactEnvLines.join('\n'));
  
  log('✅ Successfully generated React environment variables at .env.local');
  log('📊 Variables processed:');
  processedVars.forEach(v => {
    log(`   🔹 ${v.original} → ${v.react} = ${v.value}`);
  });
  
  log('🚀 Your React app is now configured to use these environment variables');
} catch (error) {
  const errorMessage = `❌ Error generating React environment variables: ${error.message}`;
  console.error(errorMessage);
  fs.appendFileSync(logFilePath, errorMessage + '\n', 'utf8');
  process.exit(1);
}