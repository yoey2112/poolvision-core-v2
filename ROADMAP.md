# Pool Vision Core v2 - Project Roadmap

## 🎯 Current Status
- ✅ Core vision system with ball detection
- ✅ Multi-camera support
- ✅ Real-time tracking and physics
- ✅ Basic game state management
- ✅ JSON output streaming

## 📋 Upcoming Features & Tasks

### Phase 1: Setup Wizard & Calibration System
**Priority: HIGH** | **Status: Not Started**

#### 1.1 Camera Setup Wizard
- [ ] Create wizard UI with step-by-step flow
- [ ] Camera selection interface
  - [ ] Display all available cameras with preview thumbnails
  - [ ] Test camera button for each device
  - [ ] Save camera preference to config
- [ ] Camera orientation selection
  - [ ] Portrait vs Landscape options
  - [ ] Rotation controls (0°, 90°, 180°, 270°)
  - [ ] Flip horizontal/vertical options
  - [ ] Live preview of transformations

#### 1.2 Table Calibration Wizard
- [ ] Interactive homography setup
  - [ ] Click to mark 4 table corners
  - [ ] Visual guides/overlay for corner selection
  - [ ] Preview of transformed view
  - [ ] Fine-tune controls for precise alignment
  - [ ] Save/load calibration profiles
- [ ] Table dimensions input
  - [ ] Standard table size presets (7ft, 8ft, 9ft)
  - [ ] Custom dimensions input
  - [ ] Metric/Imperial unit toggle
- [ ] Pocket position marking
  - [ ] Interactive pocket placement
  - [ ] Automatic pocket detection suggestion
  - [ ] Radius adjustment for each pocket
- [ ] Bird's-eye view validation
  - [ ] Show transformed top-down view
  - [ ] Grid overlay for alignment check
  - [ ] Test with ball placement

#### 1.3 Ball Detection Calibration
- [ ] Color calibration wizard
  - [ ] Place balls on table for color sampling
  - [ ] Click ball to capture color profile
  - [ ] Automatic LAB color extraction
  - [ ] Test detection with current settings
  - [ ] Adjust detection sensitivity sliders
- [ ] Detection parameter tuning
  - [ ] Min/Max radius sliders
  - [ ] Circle detection sensitivity
  - [ ] Real-time detection preview
  - [ ] Save calibration profile

### Phase 2: Graphical User Interface
**Priority: HIGH** | **Status: Not Started**

#### 2.1 Main Menu System
- [ ] Create startup screen
  - [ ] Application logo and branding
  - [ ] Animated background (pool table theme)
  - [ ] Version information
  - [ ] "Quick Start" button
- [ ] Main menu layout
  - [ ] New Game
  - [ ] Drills & Practice
  - [ ] Player Profiles
  - [ ] Analytics Dashboard
  - [ ] Settings
  - [ ] Calibration
  - [ ] Exit

#### 2.2 Settings Interface
- [ ] General Settings
  - [ ] Language selection
  - [ ] Theme (Light/Dark mode)
  - [ ] Sound effects toggle
  - [ ] Notification preferences
- [ ] Camera Settings
  - [ ] Camera selection dropdown
  - [ ] Resolution selection
  - [ ] FPS cap
  - [ ] Brightness/Contrast adjustments
  - [ ] Re-run calibration wizard
- [ ] Game Settings
  - [ ] Default game type (8-ball, 9-ball, etc.)
  - [ ] Rule variations
  - [ ] Auto-detection vs manual confirmation
  - [ ] Shot timer settings
- [ ] Display Settings
  - [ ] Fullscreen toggle
  - [ ] Window size presets
  - [ ] UI scale factor
  - [ ] Show/hide overlay elements
  - [ ] Color scheme customization

#### 2.3 UI Framework & Design System
- [ ] Design system implementation
  - [ ] Color palette definition
  - [ ] Typography system
  - [ ] Component library (buttons, cards, modals)
  - [ ] Icons and assets
  - [ ] Animation guidelines
- [ ] UI Framework selection
  - [ ] Qt integration (QML/Qt Quick)
  - [ ] Dear ImGui for immediate mode GUI
  - [ ] Custom OpenGL/DirectX overlay
  - [ ] Web-based UI (Electron/CEF)

### Phase 3: Player Profile Management
**Priority: MEDIUM** | **Status: Not Started**

#### 3.1 Player Database
- [ ] Database schema design
  - [ ] Player profiles table
  - [ ] Game sessions table
  - [ ] Shot history table
  - [ ] Statistics aggregation tables
- [ ] Database implementation
  - [ ] SQLite integration
  - [ ] CRUD operations for players
  - [ ] Data migration system
  - [ ] Backup/restore functionality

#### 3.2 Player Profile UI
- [ ] Player management screen
  - [ ] List all players with avatars
  - [ ] Add new player form
    - [ ] Name input
    - [ ] Avatar upload/selection
    - [ ] Skill level selection
    - [ ] Handedness (left/right)
    - [ ] Preferred game types
  - [ ] Edit player details
  - [ ] Delete player (with confirmation)
  - [ ] Search/filter players
- [ ] Player selection interface
  - [ ] Quick select for games
  - [ ] Recent players list
  - [ ] Guest/Anonymous mode

#### 3.3 Shot Logging System
- [ ] Shot data capture
  - [ ] Automatic shot detection
  - [ ] Shot type classification (break, bank, combo, etc.)
  - [ ] Shot outcome (success, miss, foul)
  - [ ] Ball positions before/after
  - [ ] Shot trajectory recording
  - [ ] Time taken per shot
- [ ] Shot tagging
  - [ ] Player association
  - [ ] Game session linking
  - [ ] Difficulty rating
  - [ ] Manual annotations/notes

### Phase 4: Analytics Dashboard
**Priority: MEDIUM** | **Status: Not Started**

#### 4.1 Player Statistics
- [ ] Overall statistics
  - [ ] Total games played
  - [ ] Win/Loss ratio
  - [ ] Win percentage by game type
  - [ ] Average game duration
  - [ ] Total play time
- [ ] Shot statistics
  - [ ] Total shots taken
  - [ ] Shot success rate
  - [ ] Average shot time
  - [ ] Best streak
  - [ ] Foul rate
  - [ ] Break success rate
- [ ] Advanced metrics
  - [ ] Shot difficulty analysis
  - [ ] Positional play rating
  - [ ] Consistency score
  - [ ] Improvement trends over time
  - [ ] Peak performance periods

#### 4.2 Visualization & Charts
- [ ] Chart library integration
  - [ ] Line charts for trends
  - [ ] Bar charts for comparisons
  - [ ] Pie charts for distributions
  - [ ] Heat maps for table positioning
  - [ ] Radar charts for skill profiles
- [ ] Interactive dashboards
  - [ ] Time range filters (day, week, month, year, all-time)
  - [ ] Game type filters
  - [ ] Opponent filters
  - [ ] Drill vs. match filtering
  - [ ] Export to PDF/CSV

#### 4.3 Opponent Analysis
- [ ] Head-to-head statistics
  - [ ] Win/loss vs specific opponents
  - [ ] Average margin of victory
  - [ ] Common game patterns
  - [ ] Strengths/weaknesses identified
- [ ] Opponent database
  - [ ] List of played opponents
  - [ ] Last played date
  - [ ] Overall record against each
  - [ ] Favorite opponents (most games)

#### 4.4 Game History
- [ ] Game log viewer
  - [ ] Chronological list of all games
  - [ ] Filter by player, date, type
  - [ ] Detailed game view (shot-by-shot)
  - [ ] Replay visualization
- [ ] Notable moments
  - [ ] Best games
  - [ ] Fastest wins
  - [ ] Perfect games
  - [ ] Longest streaks
  - [ ] Personal records

### Phase 5: Drill System
**Priority: MEDIUM** | **Status: Not Started**

#### 5.1 Drill Types
- [ ] Drill library
  - [ ] Breaking practice
  - [ ] Cut shots (various angles)
  - [ ] Bank shots
  - [ ] Combination shots
  - [ ] Position play exercises
  - [ ] Speed control drills
  - [ ] Rail shots
  - [ ] 9-ball run-out practice
- [ ] Custom drill creator
  - [ ] Ball position editor
  - [ ] Target definition
  - [ ] Success criteria
  - [ ] Time limits
  - [ ] Save custom drills

#### 5.2 Drill Execution
- [ ] Drill selection UI
  - [ ] Browse drill library
  - [ ] Difficulty rating
  - [ ] Preview diagram
  - [ ] Best score display
- [ ] Drill tracking
  - [ ] Success/failure detection
  - [ ] Attempts counter
  - [ ] Time tracking
  - [ ] Scoring system
  - [ ] Progress bars
- [ ] Drill completion
  - [ ] Score summary
  - [ ] Performance rating
  - [ ] Improvement suggestions
  - [ ] Save to history

### Phase 6: Match System & Enhanced UI
**Priority: HIGH** | **Status: Not Started**

#### 6.1 Match Setup
- [ ] Match creation screen
  - [ ] Player selection (vs player or vs AI)
  - [ ] Game type selection
  - [ ] Match format (race to X, best of Y)
  - [ ] Handicap settings
  - [ ] Tournament mode
- [ ] Pre-match checklist
  - [ ] Camera calibration check
  - [ ] Ball detection test
  - [ ] Player confirmation
  - [ ] Rules acknowledgment

#### 6.2 In-Game UI Layout
- [ ] Main window design (clean & crisp)
  - [ ] **Primary View** (Full camera feed)
    - [ ] High-resolution camera stream
    - [ ] Real-time ball detection overlay
    - [ ] Shot trajectory lines
    - [ ] Velocity vectors
    - [ ] Table boundaries highlighted
  
  - [ ] **Top-Right Docked Panel** (Bird's-eye view)
    - [ ] Transformed top-down table view
    - [ ] Simplified ball positions
    - [ ] Numbered ball labels
    - [ ] Pocket indicators
    - [ ] Clean minimalist design
    - [ ] Resizable/draggable
    - [ ] Toggle show/hide
  
  - [ ] **Bottom-Right Docked Panel** (Game Stats)
    - [ ] Current Score Display
      - [ ] Player 1 vs Player 2
      - [ ] Large, readable fonts
      - [ ] Color-coded by player
    - [ ] Active Player Indicator
      - [ ] Highlighted player name
      - [ ] Turn arrow/icon
      - [ ] Animated turn transition
    - [ ] Game Progress
      - [ ] Balls remaining (by group)
      - [ ] Current inning/rack
      - [ ] Time elapsed
    - [ ] Shot Information
      - [ ] Last shot result
      - [ ] Shot speed
      - [ ] Streak counter
    - [ ] Quick Stats
      - [ ] Shots taken
      - [ ] Success rate
      - [ ] Fouls committed

#### 6.3 UI Visual Design
- [ ] Modern, clean aesthetic
  - [ ] Dark theme with neon accents
  - [ ] Glass-morphism effects for panels
  - [ ] Smooth animations and transitions
  - [ ] Subtle shadows and depth
  - [ ] High contrast for readability
- [ ] Typography
  - [ ] Bold, modern fonts
  - [ ] Clear hierarchy
  - [ ] Monospace for numbers/stats
- [ ] Color scheme
  - [ ] Primary: Pool table green (#0D5E3A)
  - [ ] Accent: Neon cyan (#00FFFF) & Yellow (#FFD700)
  - [ ] Background: Dark gray (#1A1A1A)
  - [ ] Text: White (#FFFFFF) and light gray (#CCCCCC)
- [ ] Icon set
  - [ ] Custom pool-themed icons
  - [ ] Consistent style
  - [ ] Animated on hover

#### 6.4 Match Features
- [ ] Shot clock/timer
- [ ] Undo last action
- [ ] Pause/Resume match
- [ ] Save match state
- [ ] Match notes/comments
- [ ] Screenshot/recording
- [ ] Live statistics update
- [ ] Automatic foul detection
- [ ] Challenge/review system

#### 6.5 Post-Match
- [ ] Match summary screen
  - [ ] Final score
  - [ ] Match statistics
  - [ ] Shot highlights
  - [ ] MVP/best shot awards
  - [ ] Save to history
  - [ ] Share results
- [ ] Rematch option
- [ ] Return to menu

### Phase 7: Advanced Features
**Priority: LOW** | **Status: Not Started**

#### 7.1 AI Opponent
- [ ] Basic AI player
- [ ] Difficulty levels
- [ ] Shot suggestion system
- [ ] Learning from player patterns

#### 7.2 Multiplayer
- [ ] Network play support
- [ ] Online matchmaking
- [ ] Spectator mode
- [ ] Live streaming integration

#### 7.3 Tournament Mode
- [ ] Bracket management
- [ ] Tournament creation
- [ ] Automatic scheduling
- [ ] Leaderboards
- [ ] Prize tracking

#### 7.4 Video Analysis
- [ ] Automatic highlight creation
- [ ] Shot replay system
- [ ] Slow-motion analysis
- [ ] Export video clips
- [ ] YouTube integration

#### 7.5 Mobile Companion App
- [ ] Remote viewing
- [ ] Score tracking
- [ ] Push notifications
- [ ] Statistics dashboard
- [ ] Player profile sync

## 🛠️ Technical Debt & Improvements

### Code Quality
- [ ] Add comprehensive error handling
- [ ] Improve logging system
- [ ] Code documentation (Doxygen)
- [ ] Performance profiling
- [ ] Memory leak detection
- [ ] Unit test coverage > 80%

### Performance
- [ ] Optimize ball detection algorithm
- [ ] GPU acceleration for tracking
- [ ] Multi-threading improvements
- [ ] Reduce frame processing latency
- [ ] Optimize memory usage

### Compatibility
- [ ] Support more camera types
- [ ] Test on various Windows versions
- [ ] Linux compatibility testing
- [ ] macOS support (future)

## 📊 Success Metrics

- User can complete full setup in < 5 minutes
- Ball detection accuracy > 95%
- Frame processing < 16ms (60fps)
- UI response time < 100ms
- Zero crashes during normal operation
- Player satisfaction rating > 4.5/5

## 🗓️ Estimated Timeline

- **Phase 1**: 3-4 weeks
- **Phase 2**: 4-5 weeks
- **Phase 3**: 2-3 weeks
- **Phase 4**: 3-4 weeks
- **Phase 5**: 2-3 weeks
- **Phase 6**: 4-5 weeks
- **Phase 7**: 6-8 weeks (ongoing)

**Total estimated development time: 6-8 months for core features**

## 📝 Notes

- Phases can be developed in parallel by different team members
- UI/UX design should be finalized before Phase 2 implementation
- Database schema should be designed early to avoid migrations
- Regular user testing after each phase completion
- Consider accessibility features (color blind mode, screen readers)

---

**Last Updated**: November 8, 2025
**Version**: 1.0.0
