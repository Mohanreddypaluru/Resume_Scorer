# Modern Frontend - Resume Scorer

A professional, ATS-style dashboard UI for the Resume Scoring System built with Bootstrap 5.

## Features

- **Resume Upload**: Drag-and-drop or click to upload PDF, DOCX, or TXT files
- **Job Description Input**: Large textarea for pasting job descriptions
- **Score Visualization**: Dynamic score circle with color-coded results
- **Category Breakdown**: Progress bars for each scoring category
- **Skills Analysis**: Matched and missing skills with visual tags
- **Suggestions**: Actionable improvement suggestions
- **Strengths & Gaps**: Detailed analysis of resume strengths and gaps
- **Confidence & Uncertainty**: Confidence score and uncertainty indicators
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Professional ATS Style**: Clean, modern dashboard appearance

## Tech Stack

- **HTML5**: Semantic markup
- **Bootstrap 5**: UI framework and responsive grid
- **Bootstrap Icons**: Icon library
- **Custom CSS**: Professional ATS-style styling
- **Vanilla JavaScript**: No framework dependencies

## File Structure

```
modern_frontend/
├── index.html      # Main HTML structure
├── custom.css      # Custom styling
├── app.js          # JavaScript functionality
└── README.md       # This file
```

## Installation

1. Ensure the Flask backend is running on `http://localhost:5000`
2. Open `index.html` in a web browser
3. Or serve with a simple HTTP server:
```bash
cd modern_frontend
python -m http.server 3000
```
Then navigate to `http://localhost:3000`

## Usage

1. **Upload Resume**: Click the file input or drag and drop a resume file (PDF, DOCX, or TXT)
2. **Paste Job Description**: Copy and paste the job description into the textarea
3. **Click Analyze**: Click the "Analyze Resume" button to start the analysis
4. **View Results**: The results will display with:
   - Overall score (color-coded)
   - Confidence level and score
   - Category breakdown with progress bars
   - Matched and missing skills
   - Suggestions for improvement
   - Strengths and gaps analysis
   - Uncertainties and assumptions

## Design Features

### Professional ATS Style
- Clean, minimal design
- High contrast for readability
- Professional color scheme
- Card-based layout
- Clear visual hierarchy

### Responsive Design
- Mobile-first approach
- Flexible grid layout
- Touch-friendly controls
- Optimized for all screen sizes

### Visual Feedback
- Loading states with spinner
- Error messages with alerts
- Score color coding (green ≥80, blue ≥60, yellow ≥40, red <40)
- Skill tags with color coding
- Severity indicators for gaps

### Accessibility
- Semantic HTML
- ARIA labels
- Keyboard navigation
- Screen reader friendly
- High contrast ratios

## Color Scheme

- **Primary**: Blue (#0d6efd) - Main actions and highlights
- **Success**: Green (#198754) - Matched skills, high scores
- **Danger**: Red (#dc3545) - Missing skills, low scores
- **Warning**: Yellow (#ffc107) - Suggestions, partial matches
- **Info**: Cyan (#0dcaf0) - Strengths
- **Secondary**: Gray (#6c757d) - Gaps, neutral elements

## API Integration

The frontend communicates with the Flask backend via REST API:

- `POST /api/score` - Score resume file against job description
- `POST /api/score/text` - Score resume text against job description

The API base URL is configurable in `app.js` (default: `http://localhost:5000/api`).

## Customization

### Change API URL
Edit `app.js`:
```javascript
const API_BASE_URL = 'http://your-api-url/api';
```

### Modify Colors
Edit `custom.css` and update the CSS variables:
```css
:root {
    --primary-color: #your-color;
    --success-color: #your-color;
    /* etc. */
}
```

### Add New Categories
Edit `displayCategoryScores()` in `app.js` to handle new category types.

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Performance

- Minimal JavaScript (no framework overhead)
- Bootstrap CDN for fast loading
- Optimized CSS with minification ready
- Lazy loading of results
- Efficient DOM manipulation

## Security

- No sensitive data stored in browser
- File uploads processed server-side
- XSS protection via Bootstrap
- HTTPS recommended for production
