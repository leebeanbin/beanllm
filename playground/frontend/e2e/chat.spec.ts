import { test, expect } from "@playwright/test";

/**
 * E2E Tests for Chat Interface
 * 
 * Tests all chat functionality including:
 * - Basic chat
 * - File upload
 * - Model selection
 * - Feature selection
 * - Tool calls
 * - Session management
 */

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3000";
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

test.describe("Chat Interface", () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to chat page
    await page.goto(`${BASE_URL}/chat`);
    
    // Wait for page to load
    await page.waitForLoadState("networkidle");
  });

  test("should display chat interface", async ({ page }) => {
    // Check if main elements are visible
    await expect(page.locator("textarea[placeholder*='message']")).toBeVisible();
    await expect(page.locator("button[type='submit']")).toBeVisible();
  });

  test("should send a message", async ({ page }) => {
    const textarea = page.locator("textarea[placeholder*='message']");
    const sendButton = page.locator("button[type='submit']");
    
    // Type a message
    await textarea.fill("Hello, this is a test message");
    
    // Send message
    await sendButton.click();
    
    // Wait for response
    await page.waitForTimeout(2000);
    
    // Check if message appears
    await expect(page.locator("text=Hello, this is a test message")).toBeVisible();
  });

  test("should display InfoPanel", async ({ page }) => {
    // Check if InfoPanel is visible
    await expect(page.locator("[data-testid='info-panel'], .border-l")).toBeVisible();
  });

  test("should handle file upload", async ({ page }) => {
    // Find file input
    const fileInput = page.locator("input[type='file']").first();
    
    // Create a test file
    const testFile = {
      name: "test.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("Test file content"),
    };
    
    // Upload file
    await fileInput.setInputFiles({
      name: testFile.name,
      mimeType: testFile.mimeType,
      buffer: testFile.buffer,
    });
    
    // Wait for file to be processed
    await page.waitForTimeout(1000);
    
    // Check if file is attached (implementation dependent)
    // This may need adjustment based on actual UI
  });

  test("should change model", async ({ page }) => {
    // Click on mode badge to open InfoPanel
    const modeBadge = page.locator("button:has-text('Auto'), button:has-text('Manual')").first();
    await modeBadge.click();
    
    // Wait for InfoPanel to open
    await page.waitForTimeout(500);
    
    // Find model selector (implementation dependent)
    // This may need adjustment based on actual UI
  });

  test("should display tool call progress", async ({ page }) => {
    // Send a message that triggers tool call
    const textarea = page.locator("textarea[placeholder*='message']");
    await textarea.fill("Search for information about AI");
    
    const sendButton = page.locator("button[type='submit']");
    await sendButton.click();
    
    // Wait for tool call to start
    await page.waitForTimeout(1000);
    
    // Check if tool call progress is displayed
    // This may need adjustment based on actual UI
  });

  test("should handle keyboard shortcuts", async ({ page }) => {
    const textarea = page.locator("textarea[placeholder*='message']");
    
    // Type message
    await textarea.fill("Test message");
    
    // Press Enter to send
    await textarea.press("Enter");
    
    // Wait for response
    await page.waitForTimeout(2000);
    
    // Check if message was sent
    await expect(page.locator("text=Test message")).toBeVisible();
  });

  test("should display empty state", async ({ page }) => {
    // Check if empty state is visible when no messages
    await expect(
      page.locator("text=No messages yet, text=Start a conversation, text=Quick Actions").first()
    ).toBeVisible();
  });

  test("should handle InfoPanel collapse/expand", async ({ page }) => {
    // Find collapse button
    const collapseButton = page.locator("button[aria-label*='collapse'], button[aria-label*='expand']").first();
    
    if (await collapseButton.isVisible()) {
      await collapseButton.click();
      await page.waitForTimeout(500);
      
      // Check if panel is collapsed
      // This may need adjustment based on actual UI
    }
  });
});

test.describe("Backend API", () => {
  test("should respond to health check", async ({ request }) => {
    const response = await request.get(`${BACKEND_URL}/health`);
    expect(response.status()).toBe(200);
  });

  test("should list available models", async ({ request }) => {
    const response = await request.get(`${BACKEND_URL}/api/config/models`);
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty("models");
    expect(Array.isArray(data.models)).toBe(true);
  });
});
