---
title: "Old projects"
excerpt: "This post is an archive for some older projects I worked on during 2022-2023."
coverImage: ""
date: "2025-03-13"
ogImage:
  url: ""
---

This post is an archive for some older projects I worked on during 2022-2023. I didn't really write about what I was working on at the time, so I'm hoping to describe some of that here.

# Chip: a social habit-tracking platform

Chip, which is [free on the iOS App Store](https://apps.apple.com/us/app/chip-build-goals-together/id6443741171), was one of my favorite projects to work on. It's a social media platform designed to help you build and track habits. It's [built with](https://github.com/BEW111/chip) React Native for the frontend and Supabase for the backend.

I created Chip because traditional habit apps didn't seem to stick. What really kept me most accountable were my friends and family. So I designed Chip with social accountability at its core—when you record a habit (like eating healthy or working out), you take a picture and send it to your friends, like Snapchat.

I implemented a variety of features including:

- Taking and uploading pictures, or "chips", Snapchat-style
- Adding friends and seeing their new "chips" in real-time
- Highly customizable habits and goals with detailed analytics
- Shared streaks for when you and a friend share the same habit

I didn't try to advertise this app (besides to my friends) but it was fun to see it gain some natural users through the App Store!

**Takeaways from working on Chip:**

- One unexpectedly hard part was onboarding users. When you have a year to think about an idea, it's pretty hard to explain it to users in 15 seconds with no context. And if you don't explain it well, they won't bother with the app.
- I'm glad I switched from Firebase to Supabase during the project. A relational database ended up suiting me better, and Supabase also had a better dev experience and good support for cloud functions, auth, storage, etc.
- Fully publishing to the App Store is hard—there's a long and strict review process.

# Using LLMs to create knowledge graphs

This was a project I worked on at [AI For Good](https://aiforgood.itu.int/) in 2023. The goal was to take lots of unstructured documents (lots of `.pdf` and `.docx` files) and extract [knowledge graphs](https://en.wikipedia.org/wiki/Knowledge_graph) from them. These documents contained lots of domain-specific keywords and figures which were difficult for LLMs to process in their raw forms.

To turn documents into knowledge graphs, I ended up [making a fine-tune](https://huggingface.co/bew/t5_sentence_to_triplet_xl) of Google's FLAN-T5 model. The model can take in a sentence and output a `subject-relation-object` triplet, which represents an edge between two nodes in a graph. I also ended up using various computer vision libraries like OpenCV to process the figures in the documents.

# PaperRabbit

At Berkeley's 2023 LLM-themed hackathon, my friend Aagam and I built a website called PaperRabbit to help you organize your research paper reading lists.

# Milestone Monitor

To learn more about vector databases and conversational AI, my friend Shiva and I built a textable (via SMS) chatbot that helps you come up with structured goals, saves all your data, and sends you customizable reminders.
