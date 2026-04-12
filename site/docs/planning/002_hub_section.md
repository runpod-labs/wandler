# User Story: Hub Section for Wandler.ai

## Overview

**Title:** Centralized Hub for Wandler Demos and Resources  
**ID:** 002  
**Priority:** High

## User Story

**As a** developer or enthusiast interested in Wandler,  
**I want to** access a centralized hub with all demos, models, and resources,  
**So that** I can easily explore Wandler's capabilities and find the specific tools I need.

## Acceptance Criteria

1. A new `/hub` route that serves as the main entry point for all demos and resources
2. Clear navigation structure for accessing different components (demos, models, documentation)
3. `/hub/chat` route that renders the existing chat functionality from the landing page
4. Consistent UI/UX across all hub sections that follows Wandler's design system
5. Mobile-responsive design that works on all device sizes
6. Performance optimization to ensure fast loading of hub components
7. Accessible interface with proper ARIA attributes and keyboard navigation

## Technical Requirements

### Component Structure

1. **HubLayout** - Main layout component for all hub pages

   - Navigation sidebar/header
   - Content area
   - Consistent styling across all hub pages

2. **HubHome** - Main hub landing page (`/hub`)

   - Overview of available demos and resources
   - Cards/links to all available sections
   - Brief description of each component

3. **HubNavigation** - Navigation component for the hub

   - Links to all hub sections
   - Visual indication of current section
   - Collapsible on mobile

4. **DemoCard** - Component for showcasing each demo

   - Title and description
   - Visual preview/icon
   - Link to the full demo

5. **ModelCard** - Component for showcasing available models
   - Model name and description
   - Size and capability information
   - Link to demo using this model

### Implementation Details

1. **Routing Structure**

   - Use Next.js App Router with a catch-all route pattern
   - Create `/hub` as the main hub route
   - Implement a dynamic `[...slug]` route to handle all hub content pages
   - Handle `/hub/chat` and other demos through the same dynamic route handler
   - Structure the app directory as follows:
     ```
     app/
     ├── hub/
     │   ├── page.tsx           # Main hub page (/hub)
     │   ├── layout.tsx         # Shared layout for all hub pages
     │   └── [...slug]/         # Catch-all route for all demos
     │       └── page.tsx       # Handles all demos through one file
     ```

2. **Chat Integration**

   - Reuse the existing chat component from the landing page
   - Render it when the route matches `/hub/chat`
   - Ensure it functions identically when rendered in the hub context
   - Add any necessary context providers or state management

3. **UI/UX Considerations**

   - Use Shadcn components for consistent UI
   - Implement responsive design with Tailwind
   - Match the existing visual identity while providing clear navigation
   - Create smooth transitions between hub sections

4. **Performance Optimizations**

   - Implement code splitting to load only required components
   - Use Next.js static generation where possible
   - Lazy load demo components to minimize initial page load
   - Implement proper caching strategies

5. **Accessibility**
   - Ensure proper heading hierarchy
   - Add ARIA labels to all interactive elements
   - Maintain keyboard navigation throughout the hub
   - Test with screen readers

## User Flow

1. User navigates to `wandler.ai/hub`
2. User sees an overview of all available demos and resources
3. User selects the chat demo by clicking its card or navigating directly to `/hub/chat`
4. The chat interface loads, allowing the user to interact with the model
5. User can navigate back to the hub home or directly to other demo sections

## Technical Challenges

1. **Component Reusability**

   - Ensure the chat component can be used both on the landing page and in the hub
   - Manage state and context properly when components are used in different locations

2. **Consistent Experience**

   - Maintain visual and functional consistency across hub sections
   - Ensure smooth transitions between different hub components

3. **Navigation Structure**

   - Design an intuitive navigation system that scales with additional demos
   - Make navigation accessible and mobile-friendly

4. **Dynamic Routing Logic**

   - Implement clean, maintainable logic in the catch-all route handler
   - Ensure proper 404 handling for invalid routes
   - Consider server-side vs client-side routing implications

5. **Performance**
   - Optimize for quick navigation between hub sections
   - Minimize duplicate resource loading

## Dependencies

- Next.js App Router
- React
- Tailwind CSS
- Shadcn UI components
- Wandler library

## Notes

- The hub should be designed to easily accommodate future demos and resources
- Consider implementing a search function for the hub as it grows
- Analytics should track which demos are most popular
- Consider adding user preferences/settings that persist across hub sections
- Using a catch-all route provides maximum flexibility while keeping the codebase simpler
