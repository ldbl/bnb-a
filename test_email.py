#!/usr/bin/env python3
"""
Quick email test script
Run this locally with your actual Gmail app password to test email sending
"""

import os
from email_reporter import EmailReporter

def test_email_sending():
    """Test actual email sending with user credentials"""
    
    print("📧 Gmail Email Test")
    print("=" * 40)
    
    # Get credentials from user
    sender_email = input("Enter your Gmail address (stanfold@gmail.com): ").strip() or "stanfold@gmail.com"
    sender_password = input("Enter your Gmail App Password (16 chars): ").strip()
    recipient_email = input("Enter recipient email (same as sender): ").strip() or sender_email
    
    if not sender_password:
        print("❌ App password is required!")
        return False
    
    # Set environment variables
    os.environ['SENDER_EMAIL'] = sender_email
    os.environ['SENDER_PASSWORD'] = sender_password  
    os.environ['RECIPIENT_EMAIL'] = recipient_email
    
    try:
        # Initialize email reporter
        reporter = EmailReporter()
        
        # Send test email
        print(f"\n📧 Sending test email to {recipient_email}...")
        
        # Create simple test report
        test_subject = "🧪 BNB Analyzer Email Test"
        test_body = f"""
🧪 EMAIL TEST SUCCESSFUL!

Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
From: BNB Advanced Analyzer
To: {recipient_email}

✅ Email configuration is working correctly!
🚀 You can now enable daily automated reports.

Setup GitHub Secrets:
- SENDER_EMAIL: {sender_email}
- SENDER_PASSWORD: [your-app-password]
- RECIPIENT_EMAIL: {recipient_email}

Next steps:
1. Add these secrets to GitHub repository
2. Enable GitHub Actions workflow
3. Enjoy daily BNB analysis reports! 📊

---
🤖 BNB Advanced Trading Analyzer
⚠️  This is a test email
"""
        
        success = reporter.send_email(test_subject, test_body)
        
        if success:
            print("✅ Test email sent successfully!")
            print(f"📱 Check your inbox: {recipient_email}")
            print("\n🎯 Next steps:")
            print("1. Add these credentials to GitHub Secrets")
            print("2. Push the email automation code")
            print("3. Enable daily reports!")
            return True
        else:
            print("❌ Failed to send test email")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_email_sending()
