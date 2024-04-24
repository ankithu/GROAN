from dataclasses import dataclass

@dataclass
class User:
    first_name: str
    last_name: str
    email_prefix: str
    email_domain: str
    honorific: str

    def __post_init__(self):
        self.email = f"{self.email_prefix}@{self.email_domain}"
        self.full_name = f"{self.first_name} {self.last_name}"
        self.honorific_full_name = f"{self.honorific} {self.full_name}"
        self.email_honorific_full_name = f"{self.honorific} {self.email_prefix}@{self.email_domain}"
        self.name_email_combo = f"{self.full_name} <{self.email}>"

    @staticmethod
    def from_sender_string(cls, sender_string: str):
        '''
        Takes in a sender string of the format First Last <email_prefix@domain> and returns a User object.
        '''
        sender_string = sender_string.strip()
        name, email = sender_string.split('<')
        email = email.replace('>', '')
        first_name, last_name = name.split(' ')
        email_prefix, email_domain = email.split('@')
        #having 'N.o.n.e.' as the honorific shoudn't be a problem since its very
        #unlikely to be a real honorific and wont accidentally replaces any real strings in the email
        return User(first_name, last_name, email_prefix, email_domain, 'N.o.n.e.')

    def get_names(self):
        '''
        Returns a list of all the possible ways this user can be addressed.
        '''
        names = [self.name_email_combo, self.honorific_full_name, self.email_honorific_full_name, self.full_name, self.email,self.email_prefix,self.first_name,self.last_name]
        names = [name.lower() for name in names]
        return names
