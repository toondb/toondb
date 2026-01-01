#!/usr/bin/env node
/**
 * Fix ESM imports by adding .js extension to relative imports
 * This is required for Node.js ESM compatibility
 */

const fs = require('fs');
const path = require('path');

const esmDir = path.join(__dirname, '..', 'dist', 'esm');

function fixImportsInFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');
  let modified = false;

  // Fix relative imports that don't have .js extension
  // Matches: from './module' or from '../module' or import './module'
  const importRegex = /(from\s+['"])(\.\.?\/[^'"]+)(['"])/g;
  const exportRegex = /(export\s+.*\s+from\s+['"])(\.\.?\/[^'"]+)(['"])/g;

  content = content.replace(importRegex, (match, prefix, importPath, suffix) => {
    // Don't add .js if already has extension
    if (importPath.endsWith('.js') || importPath.endsWith('.json')) {
      return match;
    }
    modified = true;
    return `${prefix}${importPath}.js${suffix}`;
  });

  content = content.replace(exportRegex, (match, prefix, importPath, suffix) => {
    if (importPath.endsWith('.js') || importPath.endsWith('.json')) {
      return match;
    }
    modified = true;
    return `${prefix}${importPath}.js${suffix}`;
  });

  if (modified) {
    fs.writeFileSync(filePath, content);
    console.log(`Fixed: ${path.relative(process.cwd(), filePath)}`);
  }
}

function walkDir(dir) {
  if (!fs.existsSync(dir)) {
    console.log(`ESM directory not found: ${dir}`);
    return;
  }

  const files = fs.readdirSync(dir);
  for (const file of files) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      walkDir(filePath);
    } else if (file.endsWith('.js')) {
      fixImportsInFile(filePath);
    }
  }
}

console.log('Fixing ESM imports...');
walkDir(esmDir);
console.log('Done!');
