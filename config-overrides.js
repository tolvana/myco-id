const CopyWebpackPlugin = require('copy-webpack-plugin');
const path = require('path');

module.exports = function override(config, env) {
  // Add the CopyWebpackPlugin to the plugins array
  config.plugins.push(
    new CopyWebpackPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, 'node_modules/onnxruntime-web/dist/*.wasm'),
          to: path.resolve(__dirname, 'build/static/js/[name][ext]')
        }
      ]
    })
  );

  return config;
};
