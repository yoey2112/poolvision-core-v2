# Pool Vision Core v2 - Project Roadmap

## 🎯 Current Status
- ✅ Core vision system with ball detection
- ✅ Multi-camera support
- ✅ Real-time tracking and physics
- ✅ Basic game state management
- ✅ JSON output streaming
- ✅ **Setup Wizard - Phase 1 COMPLETE!** (Nov 8, 2025)
  - ✅ Camera selection and enumeration
  - ✅ Camera orientation controls (rotation, flip)
  - ✅ Interactive table calibration with homography
  - ✅ Table dimensions with standard presets
  - ✅ YAML configuration saving and validation
- ✅ **Main Menu & Settings - Phase 2 COMPLETE!** (Nov 8, 2025)
  - ✅ Modern main menu with animated background
  - ✅ 7 menu options with hover effects
  - ✅ Settings interface with 4 tabs
  - ✅ Complete settings persistence
- ✅ **Player Profile Management - Phase 3 COMPLETE!** (Nov 8, 2025)
  - ✅ SQLite database integration
  - ✅ Player CRUD operations (Create, Read, Update, Delete)
  - ✅ Player statistics tracking (games, wins, shots)
  - ✅ Profile UI with list/add/edit/view modes
  - ✅ Game session and shot recording system
- ✅ **Real-time Overlays - Phase 4 COMPLETE!** (Nov 8, 2025)
  - ✅ OverlayRenderer implementation with mouse interaction
  - ✅ Ball highlighting (legal/illegal indicators)
  - ✅ Shot line with power meter and direction arrows
  - ✅ Physics-based trajectory prediction with bounce points
  - ✅ Game state HUD (player turn, scores, remaining balls, fouls)
  - ✅ Ghost ball visualization for shot guidance
  - ✅ Position aids with quality heatmap
  - ✅ Real-time shot statistics and difficulty evaluation
  - ✅ Keyboard controls for overlay toggling (t/g/p/s/o)
  - ✅ UITheme integration with neon accents

## 📋 Upcoming Features & Tasks

### Phase 1: Setup Wizard & Calibration System
**Priority: HIGH** | **Status: ✅ COMPLETE** | **Completed: Nov 8, 2025**

#### 1.1 Camera Setup Wizard ✅ COMPLETE
- [x] Create wizard UI with step-by-step flow
- [x] Camera selection interface
  - [x] Display all available cameras with preview thumbnails
  - [x] Test camera button for each device
  - [x] Save camera preference to config
- [x] Camera orientation selection
  - [x] Portrait vs Landscape options
  - [x] Rotation controls (0°, 90°, 180°, 270°)
  - [x] Flip horizontal/vertical options
  - [x] Live preview of transformations

#### 1.2 Table Calibration Wizard ✅ COMPLETE
- [x] Interactive homography setup
  - [x] Click to mark 4 table corners
  - [x] Visual guides/overlay for corner selection
  - [x] Preview of transformed view
  - [x] Fine-tune controls for precise alignment
  - [ ] Save/load calibration profiles (deferred to Phase 2)
- [x] Table dimensions input
  - [x] Standard table size presets (7ft, 8ft, 9ft)
  - [x] Custom dimensions input
  - [x] Metric/Imperial unit toggle
- [ ] Pocket position marking (deferred to Phase 1.4)
  - [ ] Interactive pocket placement
  - [ ] Automatic pocket detection suggestion
  - [ ] Radius adjustment for each pocket
- [x] Bird's-eye view validation
  - [x] Show transformed top-down view
  - [ ] Grid overlay for alignment check (optional enhancement)
  - [ ] Test with ball placement (deferred)

#### 1.3 Ball Detection Calibration (DEFERRED)
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

#### 1.4 Configuration System Integration ✅ COMPLETE
- [x] YAML file generation from wizard
  - [x] Save camera config (device index, rotation, flip)
  - [x] Save table config (corners, homography, dimensions)
  - [x] Save colors config (ball color profiles)
- [x] Config validation and error handling
- [ ] Load existing configs into wizard for editing (deferred to Phase 2)

### Phase 2: Graphical User Interface
**Priority: HIGH** | **Status: ✅ COMPLETE** | **Completed: Nov 8, 2025**

#### 2.1 Main Menu System ✅ COMPLETE
- [x] Create startup screen
  - [x] Application logo and branding
  - [x] Animated background (pool table theme)
  - [x] Version information
  - [x] "Quick Start" button
- [x] Main menu layout
  - [x] New Game
  - [x] Drills & Practice
  - [x] Player Profiles
  - [x] Analytics Dashboard
  - [x] Settings
  - [x] Calibration
  - [x] Exit

#### 2.2 Settings Interface ✅ COMPLETE
- [x] General Settings
  - [x] Language selection
  - [x] Theme (Light/Dark mode)
  - [x] Sound effects toggle
  - [x] Notification preferences
- [x] Camera Settings
  - [x] Camera selection dropdown
  - [x] Resolution selection
  - [x] FPS cap
  - [x] Brightness/Contrast adjustments
  - [x] Re-run calibration wizard
- [x] Game Settings
  - [x] Default game type (8-ball, 9-ball, etc.)
  - [x] Rule variations
  - [x] Auto-detection vs manual confirmation
  - [x] Shot timer settings
- [x] Display Settings
  - [x] Fullscreen toggle
  - [x] Window size presets
  - [x] UI scale factor
  - [x] Show/hide overlay elements
  - [x] Color scheme customization

#### 2.3 UI Framework & Design System ✅ COMPLETE
- [x] Design system implementation
  - [x] Color palette definition
  - [x] Typography system
  - [x] Component library (buttons, cards, modals)
  - [x] Icons and assets
  - [x] Animation guidelines
- [x] UI Framework selection
  - [x] OpenCV-based UI (chosen for consistency)
  - [x] Modern neon-accented dark theme
  - [x] Glass-morphism effects
  - [x] Interactive hover states

### Phase 3: Player Profile Management
**Priority: MEDIUM** | **Status: ✅ COMPLETE** | **Completed: Nov 8, 2025**

#### 3.1 Player Database ✅ COMPLETE
- [x] Database schema design
  - [x] Player profiles table (id, name, avatar, skill, handedness, stats)
  - [x] Game sessions table (players, game type, winner, scores, duration)
  - [x] Shot records table (session, player, shot type, success, position, speed)
  - [x] Indexes for performance (name, player_id, session_id)
- [x] Database implementation
  - [x] SQLite3 integration via vcpkg
  - [x] CRUD operations for players (create, read, update, delete)
  - [x] Foreign key constraints with CASCADE deletes
  - [x] Statistics calculations (win rate, shot success rate)

#### 3.2 Player Profile UI ✅ COMPLETE
- [x] Player management screen
  - [x] List all players in scrollable card layout
  - [x] Add new player form
    - [x] Name input with validation
    - [x] Skill level dropdown (Beginner to Professional)
    - [x] Handedness toggle (Right/Left/Ambidextrous)
    - [x] Preferred game type dropdown
  - [x] Edit player details
  - [x] Delete player with action buttons
  - [x] Search/filter players in real-time
- [x] Player details view
  - [x] Statistics cards (games played, win rate, shot success)
  - [x] Profile information display
  - [x] Back to list navigation
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

### Phase 4: Real-time Overlays & Shot Prediction
**Priority: HIGH** | **Status: ✅ COMPLETE** | **Completed: Nov 8, 2025**

#### 4.1 OverlayRenderer System ✅ COMPLETE
- [x] Create OverlayRenderer class
  - [x] Integration with GameState and Tracker
  - [x] Mouse interaction handling
  - [x] Overlay feature flags
  - [x] Window size management
- [x] Ball highlighting
  - [x] Detected ball highlighting with number overlay
  - [x] Legal/illegal target indicators (green/red)
  - [x] Active cue ball highlighting (cyan)
  - [x] Glow effects for visibility

#### 4.2 Shot Planning Features ✅ COMPLETE
- [x] Shot line visualization
  - [x] Draw aiming line from cue ball to target
  - [x] Direction arrow indicators
  - [x] Power meter with percentage display
  - [x] Real-time updates on mouse drag
- [x] Trajectory prediction
  - [x] Physics-based ball path calculation
  - [x] Bounce point detection and visualization
  - [x] Fade-out effect for predicted path
  - [x] Collision detection with cushions
- [x] Ghost ball display
  - [x] Semi-transparent ball showing ideal contact position
  - [x] Automatic positioning based on target ball
  - [x] Visual guide for shot alignment

#### 4.3 Game State Visualization ✅ COMPLETE
- [x] Heads-up display (HUD)
  - [x] Current player turn indicator
  - [x] Score display for both players
  - [x] Remaining balls count by group
  - [x] Game state text (Open Table, Break Shot, etc.)
  - [x] Foul notifications with reason
- [x] Position aids
  - [x] Quality heatmap for position evaluation
  - [x] Legal target highlighting
  - [x] Strategic position indicators

#### 4.4 Real-time Statistics ✅ COMPLETE
- [x] Shot analysis
  - [x] Shot difficulty calculation
  - [x] Position quality evaluation
  - [x] Real-time percentage display
- [x] Shot suggestions
  - [x] Integration with GameState
  - [x] Display suggested shot positions
  - [x] Success probability indicators

#### 4.5 User Interaction ✅ COMPLETE
- [x] Keyboard controls
  - [x] 'T' - Toggle trajectory overlay
  - [x] 'G' - Toggle ghost ball
  - [x] 'P' - Toggle position aids
  - [x] 'S' - Show all statistics
  - [x] 'O' - Hide all overlays
- [x] Mouse controls
  - [x] Click and drag to aim shots
  - [x] Real-time trajectory preview
  - [x] Power adjustment based on drag distance

#### 4.6 Performance & Polish ✅ COMPLETE
- [x] UITheme integration
  - [x] Consistent neon color scheme
  - [x] Semi-transparent overlays
  - [x] Text shadows for readability
- [x] Optimization
  - [x] OpenCV optimized drawing functions
  - [x] Efficient trajectory calculation
  - [x] Minimal performance impact on detection

### Phase 5: Historical Analysis & Training
**Priority: MEDIUM** | **Status: In Progress**

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

- **Phase 1**: ~~3-4 weeks~~ → **2 days** ✅ COMPLETE (Nov 8, 2025)
  - Core wizard infrastructure: ✅ Complete
  - Config file saving: ✅ Complete
  - YAML persistence: ✅ Complete
- **Phase 2**: ~~4-5 weeks~~ → **4 hours** ✅ COMPLETE (Nov 8, 2025)
  - Main menu system: ✅ Complete
  - Settings interface: ✅ Complete
  - UI design system: ✅ Complete
- **Phase 3**: 2-3 weeks (Player profiles) → **4 hours** ✅ COMPLETE (Nov 8, 2025)
- **Phase 4**: 3-4 weeks (Analytics) → **6 hours** ✅ COMPLETE (Nov 8, 2025)
- **Phase 5**: 2-3 weeks (Historical analysis)
- **Phase 6**: 4-5 weeks (Match system with enhanced UI)
- **Phase 7**: 6-8 weeks (Advanced features - ongoing)

**Total estimated development time: 6-8 months for core features**

## 📈 Recent Progress (Nov 8, 2025)

### Phase 4: Real-time Overlays & Shot Prediction
- ✅ Implemented OverlayRenderer class with full integration
- ✅ Added ball highlighting with legal/illegal indicators
- ✅ Created shot line with power meter visualization
- ✅ Built physics-based trajectory prediction system
- ✅ Developed game state HUD with comprehensive info
- ✅ Added ghost ball visualization for shot guidance
- ✅ Implemented position aids with quality heatmap
- ✅ Created real-time shot statistics and evaluation
- ✅ Added keyboard controls (t/g/p/s/o) for overlay toggling
- ✅ Integrated with UITheme for consistent styling
- ✅ Full build and testing in table_daemon

**Files Added**: 2 new files, ~600 lines of code
- core/ui/OverlayRenderer.hpp/cpp (overlay system)
- Updated core/util/Types.hpp (BALL_RADIUS constant, helper methods)
- Updated core/track/Tracker.hpp (getBall, getBalls methods)
- Updated core/game/GameState.hpp/cpp (7 new query methods)
- Updated core/ui/UITheme.hpp/cpp (TextShadow color)
- Updated apps/table_daemon/main.cpp (overlay integration)

### Phase 3: Player Profile Management
- ✅ SQLite3 database integration via vcpkg
- ✅ Database schema with 3 tables (players, game_sessions, shot_records)
- ✅ PlayerProfile and Database classes for data management
- ✅ PlayerProfilesPage UI with list/add/edit/view modes
- ✅ Card-based player list with search functionality
- ✅ Statistics dashboard with win rate and shot success rate
- ✅ Full CRUD operations with proper foreign key constraints
- ✅ Form validation and error handling

**Files Added**: 6 new files, ~1,800 lines of code
- core/db/Database.hpp/cpp (SQLite wrapper)
- core/db/PlayerProfile.hpp/cpp (data model)
- core/ui/menu/PlayerProfilesPage.hpp/cpp (UI)

### Phase 2: Main Menu & Settings Implementation
- ✅ Created UITheme design system with modern neon-accented dark theme
- ✅ Implemented MainMenuPage with 7 interactive menu options
- ✅ Built SettingsPage with 4 tabbed sections (General, Camera, Game, Display)
- ✅ Added complete settings persistence to settings.yaml
- ✅ Created pool_vision.exe main application
- ✅ Integrated all UI components with OpenCV
- ✅ Tested menu navigation, settings tabs, and transitions
- ✅ Full build integration with CMake

**Files Added**: 8 new files, ~3,000 lines of code
- core/ui/UITheme.hpp/cpp (design system)
- core/ui/menu/MainMenuPage.hpp/cpp
- core/ui/menu/SettingsPage.hpp/cpp
- apps/pool_vision/main.cpp

### Phase 1: Setup Wizard Implementation
- ✅ Implemented complete wizard framework with WizardManager and WizardPage base classes
- ✅ Created 5 functional wizard pages (Camera Selection, Orientation, Table Calibration, Dimensions, Completion)
- ✅ Added `setup_wizard.exe` executable for guided setup
- ✅ Integrated with existing OpenCV infrastructure (no new dependencies)
- ✅ Multi-camera enumeration with live preview
- ✅ Interactive table corner selection with real-time homography transformation
- ✅ Standard table size presets and custom dimension support
- ✅ YAML configuration saving and validation
- ✅ Full build integration with CMake

**Files Added**: 17 new files, 2,429 lines of code

## 📝 Notes

- Phases can be developed in parallel by different team members
- UI/UX design should be finalized before Phase 2 implementation
- Database schema should be designed early to avoid migrations
- Regular user testing after each phase completion
- Consider accessibility features (color blind mode, screen readers)

## 🎯 Quick Start for Contributors

### Current Development Focus
Working on **Phase 1.4**: Configuration file saving system

### How to Test the Wizard
```powershell
# Build the project
cmake --build build --config Debug

# Run the setup wizard
.\build\Debug\setup_wizard.exe
```

### Project Structure
```
core/ui/               # UI components
├── WizardManager.*    # Main wizard controller
├── WizardPage.*       # Base wizard page class
└── wizard/            # Individual wizard pages
    ├── CameraSelectionPage.*
    ├── CameraOrientationPage.*
    ├── TableCalibrationPage.*
    ├── TableDimensionsPage.*
    └── CalibrationCompletePage.*

apps/setup_wizard/     # Setup wizard application entry point
```

---

**Last Updated**: November 8, 2025
**Version**: 2.0.0 (Phase 4 - COMPLETE)
