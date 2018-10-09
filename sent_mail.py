import smtplib

EMAIL_ADDRESS = "@gmail.com"
PASSWORD = ""


def send_email(subject, msg, n, TO_EMAIL_ADDRESS):
    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()

        try: 
        	server.login(EMAIL_ADDRESS,PASSWORD)
        except smtplib.SMTPAuthenticationError:
        	print("SMTPAuthenticationError")
        message = f"Subject:{subject}\n\n{msg}"
        server.sendmail(EMAIL_ADDRESS, TO_EMAIL_ADDRESS, message)
        server.quit()
        print("Success: Email sent! ",n)
    except:
        print("Email failed to send.")



def mail_function(TO_EMAIL_ADDRESS, subject, msg):
	b = "*"
	a = [b for i in range(1,len(PASSWORD))]
	a="".join(a)
	print(f"Loging in to = {EMAIL_ADDRESS}\nPassword ={a}")
	for i in range(1, 11):
		send_email(subject, msg, i, TO_EMAIL_ADDRESS)


def main():
	email = []
	n = int(input("enter the no of mails = "))
	for i in range(1,(n+1)):
		email.append(str(input()))
	subject = str(input("Subject = "))
	msg = str(input("Message = "))
	for mail in email:
		mail_function(mail, subject, msg)

if __name__ == '__main__':
	main()